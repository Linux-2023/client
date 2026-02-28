#!/usr/bin/env python3
"""
Dual-arm Piper robot deployment script with trajectory optimization.

This script deploys the VLA model for dual-arm robot control with:
- Cubic spline interpolation for smooth trajectories
- Velocity limits based on joint types
- EMA smoothing for action stability
- Separate inference (50Hz) and control (200Hz) frequencies

Usage:
    # Local model mode
    python main_dual_optimized.py --mode local --dataset_entry piper_dual --ckpt_dir /path/to/checkpoint

    # Remote server mode
    python main_dual_optimized.py --mode remote --host 0.0.0.0 --port 8000
"""
import dataclasses
import logging
import pathlib
import signal
import sys
import os
import threading
import time

import numpy as np
from scipy.interpolate import CubicSpline

from env_dual import PiperDualEnvironment
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro


# ==================== 轨迹优化参数 ====================
INFERENCE_FREQ = 30   # 推理频率 30Hz
CONTROL_FREQ = 200    # 控制频率 200Hz
INTERPOLATION_STEPS = CONTROL_FREQ // INFERENCE_FREQ  # 每次推理之间的插值步数 (6)

# EMA平滑参数
EMA_ALPHA = 0.3


# ==================== 轨迹优化函数 ====================
def interpolate_actions(prev_action: np.ndarray, next_action: np.ndarray, steps: int) -> np.ndarray:
    """使用三次样条插值在两个动作之间进行平滑插值 (吕国栋版本)
    
    Args:
        prev_action: 前一个动作
        next_action: 下一个动作 
        steps: 插值步数
        
    Returns:
        ndarray: 插值后的动作序列，形状为(steps, action_dim)
    """
    # 创建时间点（t=0, 0.3, 0.7, 1）以实现缓入缓出
    t = np.array([0, 0.3, 0.7, 1])
    
    # 创建控制点（添加中间控制点以增加平滑度）
    mid_point1 = prev_action * 0.7 + next_action * 0.3
    mid_point2 = prev_action * 0.3 + next_action * 0.7
    actions = np.vstack([prev_action, mid_point1, mid_point2, next_action])
    
    # 为每个维度创建三次样条插值器
    try:
        cs = CubicSpline(t, actions, axis=0, bc_type='natural')
    except ValueError as e:
        # 极端情况下（例如 steps < 2），CubicSpline 可能失败，回退到线性插值
        print(f"[WARNING] CubicSpline Error: {e}. Fallback to linear interpolation.")
        return np.linspace(prev_action, next_action, steps)
    
    # 生成均匀分布的时间点
    t_interp = np.linspace(0, 1, steps)
    
    # 计算插值点
    interpolated = cs(t_interp)
    
    # 确保起点和终点严格等于输入值
    interpolated[0] = prev_action
    interpolated[-1] = next_action
    
    return interpolated


def calculate_velocity_limits(action_diff: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """根据不同关节类型设置速度限制 (吕国栋版本)
    
    Args:
        action_diff: 动作差值
        prev_action: 前一个动作(用于确定关节类型)
    
    Returns:
        修正后的动作差值
    """
    # 1. 将 deg/s 转换为 rad/s
    # J1: 180, J2: 195, J3: 180, J4: 225, J5: 225, J6: 225
    deg_per_sec_limits = np.array([180.0, 195.0, 180.0, 225.0, 225.0, 225.0])
    rad_per_sec_limits = deg_per_sec_limits * (np.pi / 180.0)
    
    # 2. 计算每个推理周期的最大变化量 (delta)
    # 推理周期 = 1.0 / INFERENCE_FREQ (即 1.0 / 50 = 0.02s)
    inference_cycle_duration = 1.0 / INFERENCE_FREQ
    max_delta_per_cycle = rad_per_sec_limits * inference_cycle_duration
    
    # 3. 创建14维的速度限制向量 (假设 prev_action 维度为 14)
    max_velocity = np.zeros(prev_action.shape[0])
    
    # 左臂设置
    max_velocity[0:6] = max_delta_per_cycle  # J1-J6
    max_velocity[6] = float('inf')           # 左臂夹持器 (无限制)
    
    # 右臂设置 (假设与左臂具有相同的速度限制)
    max_velocity[7:13] = max_delta_per_cycle # J1-J6
    max_velocity[13] = float('inf')          # 右臂夹持器 (无限制)
    
    # 4. 计算缩放因子
    velocity = np.abs(action_diff)
    # 避免除以零
    scale = np.minimum(1.0, max_velocity / (velocity + 1e-9)) 
    
    return action_diff * scale


@dataclasses.dataclass
class Args:
    """Arguments for dual-arm deployment with trajectory optimization."""
    
    # Output settings
    out_dir: pathlib.Path = pathlib.Path("data/piper_dual/videos")
    
    # Basic settings
    seed: int = 0
    
    # Frequency settings (trajectory optimization)
    inference_freq: int = 30    # 推理频率 Hz
    control_freq: int = 200     # 控制频率 Hz
    camera_fps: int = 30        # 相机帧率 Hz (RealSense 通常支持 30Hz)
    
    # EMA smoothing
    ema_alpha: float = 0.3      # EMA平滑系数
    
    # Action settings
    action_horizon: int = 15    # 每次推理后执行的动作数量
    max_action_horizon: int = 32  # 模型输出的最大动作步数
    
    # Episode settings
    num_episodes: int = 1
    max_steps: int = 8000
    
    # Mode selection: "local" or "remote"
    mode: str = "remote"
    
    # Remote server settings (for mode="remote")
    host: str = "127.0.0.1"
    port: int = 8000

    max_pad_state_and_action_length: int = 64
    device: str = "cuda:0"
    
    # Display settings
    display: bool = False
    
    # Camera settings
    high_camera_id: str = "148522073709"  # RealSense serial number
    left_wrist_camera_id: int = 0
    right_wrist_camera_id: int = 8
    
    # CAN bus settings
    left_can_port: str = "can_left"
    right_can_port: str = "can_right"
    
    # Task settings
    prompt: str = "Pick up anything on the table and put it in the basket."
    
    # Control settings
    gripper_norm: bool = True
    tele_mode: bool = False
    record_mode: bool = False
    
    # Video recording
    save_video: bool = False


class TrajectoryOptimizedController:
    """轨迹优化控制器，集成EMA平滑、速度限制和样条插值
    
    支持 Action Chunk 缓存和逐步消费
    """
    
    def __init__(
        self,
        inference_freq: int = INFERENCE_FREQ,
        control_freq: int = CONTROL_FREQ,
        ema_alpha: float = EMA_ALPHA,
        action_horizon: int = 15,  # 每次推理后执行的动作数量
    ):
        self.inference_freq = inference_freq
        self.control_freq = control_freq
        self.interpolation_steps = control_freq // inference_freq
        self.ema_alpha = ema_alpha
        self.action_horizon = action_horizon
        
        # 状态变量
        self.smoothed_action = None
        self.prev_action = None
        self.step = 0
        
        # Action Chunk 缓存
        self.action_chunk = None  # 存储完整的 action chunk (horizon, action_dim)
        self.chunk_index = 0      # 当前消费到的位置
        
    def reset(self):
        """重置控制器状态"""
        self.smoothed_action = None
        self.prev_action = None
        self.step = 0
        self.action_chunk = None
        self.chunk_index = 0
        
    def initialize_from_state(self, state: np.ndarray):
        """从当前状态初始化控制器
        
        Args:
            state: 当前机器人状态 (14维: 左臂6关节+1夹爪, 右臂6关节+1夹爪)
        """
        self.prev_action = np.asarray(state, dtype=np.float64).copy()
        self.smoothed_action = self.prev_action.copy()
    
    def needs_new_chunk(self) -> bool:
        """检查是否需要新的 action chunk"""
        if self.action_chunk is None:
            return True
        if self.chunk_index >= min(self.action_horizon, len(self.action_chunk)):
            return True
        return False
    
    def update_action_chunk(self, action_chunk: np.ndarray):
        """更新 action chunk 缓存
        
        Args:
            action_chunk: 模型输出的动作序列，形状为 (horizon, action_dim) 或 (action_dim,)
        """
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        
        # 如果是单个动作，扩展为 (1, action_dim)
        if len(action_chunk.shape) == 1:
            action_chunk = action_chunk.reshape(1, -1)
        
        self.action_chunk = action_chunk
        self.chunk_index = 0
        
    def get_next_action(self) -> np.ndarray:
        """从缓存中获取下一个原始动作
        
        Returns:
            下一个动作 (action_dim,)
        """
        if self.action_chunk is None:
            raise RuntimeError("Action chunk 未初始化，请先调用 update_action_chunk")
        
        if self.chunk_index >= len(self.action_chunk):
            # 如果超出范围，返回最后一个动作
            action = self.action_chunk[-1]
        else:
            action = self.action_chunk[self.chunk_index]
            self.chunk_index += 1
        
        return action
        
    def process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """处理原始动作，返回插值后的动作序列
        
        Args:
            raw_action: 单个动作 (action_dim,)
            
        Returns:
            interpolated_actions: 插值后的动作序列，形状为(interpolation_steps, action_dim)
        """
        # 确保是 numpy 数组
        raw_action = np.asarray(raw_action, dtype=np.float64)
        
        # 1. EMA平滑
        if self.smoothed_action is None:
            self.smoothed_action = raw_action.copy()
        else:
            self.smoothed_action = (self.ema_alpha * raw_action) + ((1 - self.ema_alpha) * self.smoothed_action)
        
        # 2. 目标动作是平滑后的动作
        next_action_target = self.smoothed_action.copy()
        
        # 3. 如果是第一步，使用当前状态作为起点
        if self.prev_action is None:
            self.prev_action = raw_action.copy()
            return np.tile(raw_action, (self.interpolation_steps, 1))
        
        # 4. 计算动作差值并应用速度限制
        action_diff = next_action_target - self.prev_action
        limited_diff = calculate_velocity_limits(action_diff, self.prev_action)
        
        # 5. 计算本周期内插值的最终目标点
        final_target_action = self.prev_action + limited_diff
        
        # 6. 在两次推理之间进行三次样条插值
        interpolated_actions = interpolate_actions(
            self.prev_action, 
            final_target_action, 
            self.interpolation_steps
        )
        
        # 7. 更新前一个动作
        self.prev_action = final_target_action.copy()
        self.step += 1
        
        return interpolated_actions


def main(args: Args) -> None:
    environment = None
    emergency_shutdown = threading.Event()
    
    # 更新全局频率参数
    global INFERENCE_FREQ, CONTROL_FREQ, INTERPOLATION_STEPS
    INFERENCE_FREQ = args.inference_freq
    CONTROL_FREQ = args.control_freq
    INTERPOLATION_STEPS = CONTROL_FREQ // INFERENCE_FREQ
    
    # Signal handler for safe shutdown
    def signal_handler(sig, frame):
        print("\n🛑 检测到 Ctrl+C，正在安全关闭...")
        emergency_shutdown.set()
        try:
            if environment:
                environment.close()
        except Exception as e:
            print(f"❌ 关闭环境时出错: {e}")
        finally:
            print("✅ 程序已退出")
            os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("=" * 60)
        print("🤖 双臂 Piper 机器人 - 轨迹优化部署")
        print("=" * 60)
        print(f"📊 推理频率: {args.inference_freq} Hz")
        print(f"📊 控制频率: {args.control_freq} Hz")
        print(f"📊 相机帧率: {args.camera_fps} Hz")
        print(f"📊 插值步数: {INTERPOLATION_STEPS}")
        print(f"📊 EMA系数: {args.ema_alpha}")
        
        # Initialize dual-arm environment
        print("\n🚀 正在初始化双臂 Piper 环境...")
        environment = PiperDualEnvironment(
            left_can_port=args.left_can_port,
            right_can_port=args.right_can_port,
            high_camera_id=args.high_camera_id,
            left_wrist_camera_id=args.left_wrist_camera_id,
            right_wrist_camera_id=args.right_wrist_camera_id,
            camera_fps=args.camera_fps,  # 使用相机帧率参数
            seed=args.seed,
            tele_mode=args.tele_mode,
            prompt=args.prompt,
            max_episode_steps=args.max_steps,
            watchdog_timeout=5.0,
            show_usb_camera=args.display,
            gripper_norm=args.gripper_norm,
            record_mode=args.record_mode,
        )
        print("✅ 双臂 Piper 环境初始化成功")
        print(f"   - 左臂 CAN: {args.left_can_port}")
        print(f"   - 右臂 CAN: {args.right_can_port}")
        print(f"   - 全局相机: {args.high_camera_id}")
        print(f"   - 左腕相机: {args.left_wrist_camera_id}")
        print(f"   - 右腕相机: {args.right_wrist_camera_id}")
        
        # Initialize policy based on mode
        print(f"\n🚀 正在初始化策略 (模式: {args.mode})...")
        
        if args.mode == "local":
            # Local model inference
            if not args.ckpt_dir:
                raise ValueError("使用 local 模式时必须指定 --ckpt_dir 参数")
            
            from openpi_client.zr0_policy import ZR0LocalPolicy
            
            policy = ZR0LocalPolicy(
                dataset_entry=args.dataset_entry,
                ckpt_dir=args.ckpt_dir,
                window_size=args.window_size,
                use_ecot=args.use_ecot,
                num_denoised_steps=args.num_denoised_steps,
                max_pad_state_and_action_length=args.max_pad_state_and_action_length,
                device=args.device,
                action_horizon=args.action_horizon,
            )
            print("✅ ZR-0 本地模型加载成功")
            print(f"   - 模型路径: {args.ckpt_dir}")
            print(f"   - 数据集: {args.dataset_entry}")
            
        elif args.mode == "remote":
            # Remote websocket server
            policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            )
            print("✅ 远程策略服务器连接成功")
            print(f"   - 服务器地址: {args.host}:{args.port}")
            
        else:
            raise ValueError(f"未知模式: {args.mode}，支持的模式: local, remote")
        
        # Initialize trajectory controller
        trajectory_controller = TrajectoryOptimizedController(
            inference_freq=args.inference_freq,
            control_freq=args.control_freq,
            ema_alpha=args.ema_alpha,
            action_horizon=args.action_horizon,
        )
        
        print("\n" + "=" * 60)
        print(f"🎯 任务: {args.prompt}")
        print("=" * 60)
        
        # Video writer setup
        video_writer = None
        if args.save_video:
            import cv2
            video_path = args.out_dir / f"episode_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            args.out_dir.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, args.control_freq, (640, 480))
            print(f"📹 视频保存路径: {video_path}")
        
        # Run episodes
        for episode_idx in range(args.num_episodes):
            print(f"\n🎬 开始第 {episode_idx + 1}/{args.num_episodes} 个 Episode...")
            
            # Reset environment and controller
            environment.reset()
            trajectory_controller.reset()
            
            # Get initial observation after reset
            obs = environment.get_observation()
            
            # Get initial state
            state = obs.get("observation.state", None)
            if state is not None:
                trajectory_controller.initialize_from_state(state)
            
            step = 0
            episode_start_time = time.time()
            
            print("\n🏃 开始运行策略... (按 Ctrl+C 停止)")
            print(f"   - 动作执行步数 (action_horizon): {args.action_horizon}")
            print(f"   - 每个动作插值步数: {trajectory_controller.interpolation_steps}")
            
            while step < args.max_steps and not emergency_shutdown.is_set():
                
                # 检查是否需要新的推理（action chunk 用完了）
                if trajectory_controller.needs_new_chunk():
                    inference_start_time = time.time()
                    
                    # Get observation
                    obs = environment.get_observation()
                    
                    # Build policy input (使用 ZR-0 期望的键名格式)
                    policy_input = {
                        "prompt": args.prompt,
                        "observation.state": obs.get("observation.state"),
                        "task": obs.get("task", args.prompt),
                        "n_action_steps": args.max_action_horizon,
                    }
                    
                    # 添加图像数据
                    images = obs.get("images", {})
                    for cam_name, img in images.items():
                        policy_input[f"observation.images.{cam_name}"] = img
                    
                    # Get action chunk from policy
                    try:
                        result = policy.infer(policy_input)
                        action_chunk = result.get("actions", result.get("action"))
                        action_chunk = np.asarray(action_chunk, dtype=np.float64)
                        
                        # 确保是 2D 数组 (horizon, action_dim)
                        if len(action_chunk.shape) == 1:
                            action_chunk = action_chunk.reshape(1, -1)
                        
                        # 更新 action chunk 缓存
                        trajectory_controller.update_action_chunk(action_chunk)
                        
                        inference_elapsed = time.time() - inference_start_time
                        if step % 50 == 0:
                            print(f"   🔄 新推理: 获取 {len(action_chunk)} 步动作, 推理耗时: {inference_elapsed*1000:.1f}ms")
                        
                    except Exception as e:
                        print(f"❌ 推理错误: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 从缓存中获取下一个动作
                raw_action = trajectory_controller.get_next_action()
                
                # Process action through trajectory controller (EMA平滑 + 速度限制 + 插值)
                interpolated_actions = trajectory_controller.process_action(raw_action)
                
                # Execute interpolated actions at control frequency
                for interp_idx, interp_action in enumerate(interpolated_actions):
                    action_start_time = time.time()
                    
                    if emergency_shutdown.is_set():
                        break
                    
                    # Execute action (使用 apply_action 方法，传入字典格式)
                    environment.apply_action({"actions": interp_action})
                    
                    # Record video frame
                    if video_writer is not None:
                        images = obs.get("images", {})
                        frame = images.get("cam_high")
                        if frame is not None:
                            import cv2
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(frame_bgr)
                    
                    # Control execution frequency
                    elapsed = time.time() - action_start_time
                    sleep_time = 1.0 / args.control_freq - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Log progress
                if step % 100 == 0:
                    elapsed_time = time.time() - episode_start_time
                    actual_freq = step / elapsed_time if elapsed_time > 0 else 0
                    print(f"   📍 Step {step}/{args.max_steps}, chunk_idx: {trajectory_controller.chunk_index}/{args.action_horizon}, 执行频率: {actual_freq:.1f} Hz")
                
                step += 1
            
            episode_duration = time.time() - episode_start_time
            print(f"\n✅ Episode {episode_idx + 1} 完成")
            print(f"   - 总步数: {step}")
            print(f"   - 耗时: {episode_duration:.1f} 秒")
            print(f"   - 平均推理频率: {step / episode_duration:.1f} Hz")
        
        if video_writer is not None:
            video_writer.release()
            print(f"📹 视频已保存")
        
    except RuntimeError as e:
        error_msg = str(e)
        if "看门狗超时" in error_msg or "不健康状态" in error_msg or "断开连接" in error_msg:
            print(f"\n{'='*60}")
            print("🚨 检测到硬件设备故障！")
            print(f"{'='*60}")
            print(f"错误信息: {e}")
            print("\n可能的原因:")
            print("  1. USB 相机（腕部相机或全局相机）突然断开连接")
            print("  2. CAN 总线连接中断，机械臂失去通信")
            print("  3. USB 带宽不足或供电不稳定")
            print("  4. 驱动程序卡死")
            print("\n建议的解决方案:")
            print("  1. 检查所有 USB 连接是否牢固")
            print("  2. 检查 CAN 总线连接")
            print("  3. 尝试更换 USB 端口")
            print("  4. 检查电源供应是否充足")
            print("  5. 重启机械臂和相机")
            print(f"{'='*60}\n")
        else:
            print(f"\n❌ 运行时错误: {e}")
            import traceback
            traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n🛑 检测到 Ctrl+C，正在安全关闭...")
    
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🔄 正在清理资源...")
        if environment:
            try:
                environment.close()
            except Exception as e:
                print(f"❌ 关闭环境时出错: {e}")
        
        print("✅ 程序已完成")
        time.sleep(0.5)
        os._exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
