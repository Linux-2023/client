#!/usr/bin/env python3
"""
双臂 Piper 机器人动作复现脚本 (无插值版，适配 env_dual.py)。

Usage:
- 指定HDF5数据文件路径运行脚本，机器人会直接跳转到记录的关节位置
- 按 'q' 退出程序，按 'p' 暂停/继续复现

注意：去掉插值后，机器人动作可能会比较生硬，请确保播放速度不要过快以免损坏电机。
"""
import time
import numpy as np
import cv2
import h5py
import os
import argparse
from typing import Dict, List, Optional

# 导入正确的环境类
from env_dual import PiperDualEnvironment

def load_hdf5_episode(file_path: str) -> Dict:
    """
    从HDF5文件加载录制的episode数据
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    episode_data = {}
    
    with h5py.File(file_path, "r") as f:
        # 加载关节位置数据
        if "observations/qpos" in f:
            episode_data["qpos"] = f["observations/qpos"][:]
        else:
            raise KeyError("HDF5文件中未找到 'observations/qpos' 数据集")
        
        # 加载动作数据 (通常是下一时刻的qpos)
        if "action" in f:
            episode_data["action"] = f["action"][:]
        else:
            print("警告: 未找到 'action' 数据集，将使用 'qpos' 代替")
            episode_data["action"] = episode_data["qpos"]
        
        # 加载图像数据
        episode_data["images"] = {}
        if "observations/images" in f:
            img_group = f["observations/images"]
            for cam_name in img_group.keys():
                episode_data["images"][cam_name] = img_group[cam_name][:]
        
        # 加载任务描述
        if "task" in f:
            episode_data["task"] = f["task"][:].astype(str)
            if isinstance(episode_data["task"][0], bytes):
                episode_data["task"] = [t.decode('utf-8') for t in episode_data["task"]]
        else:
            episode_data["task"] = ["Unknown task"] * len(episode_data["qpos"])
        
        episode_data["num_steps"] = len(episode_data["qpos"])
        
        print(f"成功加载HDF5文件: {file_path}")
        print(f"  总步数: {episode_data['num_steps']}")
        print(f"  关节状态维度: {episode_data['qpos'].shape[1]}")
        print(f"  任务描述: {episode_data['task'][0]}")
    
    return episode_data

def display_replay(step_idx: int, episode_data: Dict, window_name: str = "Replay", paused: bool = False) -> int:
    """
    显示复现过程中的图像和状态信息
    """
    display_images = []
    images_dict = episode_data["images"]
    
    for cam_name, img_sequence in images_dict.items():
        if step_idx < len(img_sequence):
            img = img_sequence[step_idx]
            
            # 格式转换
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                img_hwc = img

            if img_hwc.shape[2] == 1:
                img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_GRAY2BGR)
            elif img_hwc.shape[2] == 3:
                img_bgr = img_hwc[:, :, ::-1].copy()
            else:
                continue

            if img_bgr.dtype != np.uint8:
                if img_bgr.max() <= 1.0:
                    img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
                else:
                    img_bgr = img_bgr.astype(np.uint8)
            
            # 调整大小
            target_size = 448
            h, w, _ = img_bgr.shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_bgr, (new_w, new_h))
            
            start_h = (new_h - target_size) // 2
            start_w = (new_w - target_size) // 2
            img_cropped = img_resized[start_h:start_h+target_size, start_w:start_w+target_size]
            
            # 添加标签
            label_color = (0, 0, 255) if paused else (0, 255, 0)
            status_text = "PAUSED" if paused else "PLAYING"
            label_text = f"{cam_name} | {status_text}"
            cv2.putText(img_cropped, label_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
            
            display_images.append(img_cropped)
    
    if not display_images:
        placeholder = np.zeros((224, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No camera data / Visualization Only", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        display_images = [placeholder]
    
    combined = np.hstack(display_images) if len(display_images) > 1 else display_images[0]
    
    # 添加底部信息栏
    banner_height = 60
    banner = np.zeros((banner_height, combined.shape[1], 3), dtype=np.uint8)
    
    info_texts = [
        f"Step: {step_idx + 1}/{episode_data['num_steps']}",
        f"Mode: Direct Position Control (No Interpolation)",
        f"Task: {episode_data['task'][min(step_idx, len(episode_data['task'])-1)]}"
    ]
    
    y_offset = 20
    for text in info_texts:
        cv2.putText(banner, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
    combined = np.vstack([combined, banner])
    
    cv2.imshow(window_name, combined)
    return cv2.waitKey(1) & 0xFF

def parse_args():
    parser = argparse.ArgumentParser(description="Piper机器人动作复现脚本 (无插值，适配env_dual.py)")
    
    parser.add_argument(
        "--hdf5_file",
        type=str,
        required=True,
        help="要复现的HDF5数据文件路径"
    )
    
    parser.add_argument(
        "--left_can_port", type=str, default="can_left", help="左臂CAN端口"
    )
    parser.add_argument(
        "--right_can_port", type=str, default="can_right", help="右臂CAN端口"
    )
    
    parser.add_argument(
        "--camera_fps", type=int, default=30, help="相机帧率"
    )
    
    parser.add_argument(
        "--high_camera_id", type=str, default=148522073709, help="全局相机ID"
    )
    parser.add_argument(
        "--left_wrist_camera_id", type=int, default=0, help="左手腕相机ID"
    )
    parser.add_argument(
        "--right_wrist_camera_id", type=int, default=9, help="右手腕相机ID"
    )
    
    parser.add_argument(
        "--step_delay",
        type=float,
        default=0.05,
        help="每步之间的延迟时间（秒）。由于去掉了插值，建议设置为0.05-0.1以保证平滑。默认: 0.05"
    )
    
    parser.add_argument(
        "--action_velocity",
        type=int,
        default=30,
        help="机械臂运动速度（默认30，范围10-50，值越小越慢）"
    )
    
    parser.add_argument(
        "--skip_robot_control",
        action="store_true",
        help="仅播放视觉数据，不控制机器人"
    )
    
    return parser.parse_args()

def control_robot_joints(env: PiperDualEnvironment, target_joint_pos: np.ndarray, velocity: int = 30):
    """
    适配 env_dual.py 的机械臂控制函数
    使用 apply_action 方法发送关节位置指令
    
    Args:
        env: PiperDualEnvironment 实例
        target_joint_pos: 14维关节位置数组 [左臂7维, 右臂7维]
        velocity: 机械臂运动速度 (10-50)
    """
    # 检查关节位置维度是否正确（14维：左臂7关节+右臂7关节）
    if len(target_joint_pos) != 14:
        raise ValueError(f"关节位置维度错误，期望14维，实际{len(target_joint_pos)}维")
    
    # 构造 env_dual.py 期望的 action 字典格式
    action = {
        "actions": target_joint_pos.tolist()  # 转换为list，确保格式兼容
    }
    
    # 临时修改机器人控制速度（如果需要）
    # 注意：env_dual.py 中 apply_action 内部调用 control_dual_joint 时固定了 velocity=30
    # 如果需要自定义速度，需要确保 PiperDualController 的 control_dual_joint 支持传入 velocity
    try:
        # 发送动作指令
        env.apply_action(action)
        # 可选：直接调用底层控制器（如果 apply_action 封装了额外逻辑）
        # env._robot.control_dual_joint(target_joint_pos.tolist(), velocity=velocity)
    except Exception as e:
        print(f"机械臂控制失败: {e}")
        raise

def get_current_joint_pos(env: PiperDualEnvironment) -> np.ndarray:
    """
    获取当前机械臂关节位置
    
    Args:
        env: PiperDualEnvironment 实例
    
    Returns:
        14维关节位置数组
    """
    try:
        # 通过 get_observation 获取当前状态
        obs = env.get_observation()
        current_pos = obs["observation.state"]
        return current_pos
    except Exception as e:
        print(f"获取当前关节位置失败: {e}")
        # 返回全零数组作为备用
        return np.zeros(14, dtype=np.float32)

def main():
    args = parse_args()
    
    # 1. 加载数据
    try:
        episode_data = load_hdf5_episode(args.hdf5_file)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    # 2. 初始化环境
    print("\n=== 初始化机器人环境 ===")
    # 注意：tele_mode=False 以便发送指令
    env = PiperDualEnvironment(
        left_can_port=args.left_can_port,
        right_can_port=args.right_can_port,
        camera_fps=args.camera_fps,
        tele_mode=False,  # 必须设为False才能控制机器人
        high_camera_id=args.high_camera_id,
        left_wrist_camera_id=args.left_wrist_camera_id,
        right_wrist_camera_id=args.right_wrist_camera_id,
        prompt=episode_data["task"][0],
        show_usb_camera=False  # 复现时不显示额外的USB相机窗口
    )
    
    # 重置环境（使能机器人）
    env.reset()
    
    cv2.namedWindow("Replay", cv2.WINDOW_AUTOSIZE)
    
    # 3. 复现循环
    step_idx = 0
    paused = False
    num_steps = episode_data["num_steps"]
    
    print("\n=== 开始复现 ===")
    print(f"每步延迟: {args.step_delay}秒")
    print(f"机械臂运动速度: {args.action_velocity}")
    print("按键说明: 'q'退出, 'p'暂停")
    
    try:
        # 初始归位（移动到第一步的位置）
        if not args.skip_robot_control:
            print("正在移动到初始位置...")
            initial_qpos = episode_data["qpos"][0]
            control_robot_joints(env, initial_qpos, args.action_velocity)
            time.sleep(2.0)  # 等待机器人到达初始位置

        while step_idx < num_steps:
            # 显示画面
            key = display_replay(step_idx, episode_data, paused=paused)
            
            # 按键处理
            if key == ord('q') or key == 27:
                print("用户退出")
                break
            elif key == ord('p'):
                paused = not paused
                time.sleep(0.2)  # 防止按键连点
                if paused:
                    print("已暂停")
                else:
                    print("继续播放")

            if not paused:
                # 4. 核心逻辑：直接控制
                if not args.skip_robot_control:
                    # 获取当前帧对应的目标关节位置
                    # 优先使用action，如果action维度不对则使用qpos
                    if step_idx < len(episode_data["action"]):
                        target_qpos = episode_data["action"][step_idx]
                    else:
                        print(f"警告: 第{step_idx}步缺少动作数据，使用关节位置代替")
                        break
                    
                    # 确保目标位置是14维
                    if len(target_qpos) != 14:
                        print(f"警告: 第{step_idx}步关节位置维度错误，跳过该步")
                        step_idx += 1
                        time.sleep(args.step_delay)
                        continue
                    
                    # 发送关节控制指令
                    control_robot_joints(env, target_qpos, args.action_velocity)
                
                # 控制播放速度
                time.sleep(args.step_delay)
                
                step_idx += 1
                if step_idx % 50 == 0:
                    print(f"进度: {step_idx}/{num_steps}")

    except KeyboardInterrupt:
        print("\n程序中断")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== 结束 ===")
        # 关闭环境（自动失能机器人、关闭相机）
        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()