#!/usr/bin/env python3
"""
Dual-arm Piper robot deployment script with ZR-0/PI05 models.

This script deploys the ZR-0 and PI05 VLA models for dual-arm robot control.
It supports remote websocket server modes.

Usage:
    # Remote server mode (connects to a remote inference server)
    python main_dual.py --mode remote --host 0.0.0.0 --port 8000
"""
import dataclasses
import logging
import pathlib
import signal
import sys
import os
import threading
import time

from env_dual import PiperDualEnvironment
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
from plot_dynamics import RobotStatePlotter
import tyro


@dataclasses.dataclass
class Args:
    """Arguments for dual-arm ZR-0/PI05 deployment."""
    
    # Output settings
    out_dir: pathlib.Path = pathlib.Path("data/piper_dual/videos")
    
    # Basic settings
    seed: int = 0
    
    # Action settings
    max_action_horizon: int = 50
    action_horizon: int = 10
    fps: int = 30
    actions_during_latency: int = 5
    num_steps: int = 800
    num_episodes: int = 1
    run_tag: str = ""  # 用于记录实验标签，如 "towel_base_RTC"

    
    # Mode selection: "local" or "remote"
    mode: str = "remote"
    
    # Remote server settings (for mode="remote")
    # host: str = "0.0.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    
    
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
    prompt: str = "Fold_the_towel"
    # prompt: str = "Pick up anything on the table and put it in the basket."
    
    # Control settings
    use_async: bool = True
    use_rtc: bool = False
    gripper_norm: bool = True
    tele_mode: bool = False
    record_mode: bool = True


def main(args: Args) -> None:
    environment = None
    runtime = None
    emergency_shutdown = threading.Event()
    
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
        print("🤖 双臂 Piper 机器人 - ZR-0/PI05 部署")
        print("=" * 60)
        
        # Initialize dual-arm environment
        print("\n🚀 正在初始化双臂 Piper 环境...")
        environment = PiperDualEnvironment(
            left_can_port=args.left_can_port,
            right_can_port=args.right_can_port,
            high_camera_id=args.high_camera_id,
            left_wrist_camera_id=args.left_wrist_camera_id,
            right_wrist_camera_id=args.right_wrist_camera_id,
            camera_fps=args.fps,
            seed=args.seed,
            tele_mode=args.tele_mode,
            prompt=args.prompt,
            max_episode_steps=args.num_steps,
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
            
        if args.mode == "remote":
            # Remote websocket server
            base_policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
            )
            print("✅ 远程策略服务器连接成功")
            print(f"   - 服务器地址: {args.host}:{args.port}")
            
        else:
            raise ValueError(f"未知模式: {args.mode}，支持的模式: remote")
        
        # Wrap with action chunk broker
        print("\n🚀 正在初始化运行时...")
        if args.use_async:
            policy = action_chunk_broker.ActionChunkBroker_RTC(
                policy=base_policy,
                action_horizon=args.action_horizon,
                fps=args.fps,
                actions_during_latency=args.actions_during_latency,
                use_rtc=args.use_rtc,
            )
            print(policy._action_horizon, policy._max_horizon)
        else:
            policy = action_chunk_broker.ActionChunkBroker(
                policy=base_policy,
                action_horizon=args.action_horizon,
                fps=args.fps,
            )
        
        # 将 broker 传给 plotter，异步模式下可获取真实 chunk 边界
        broker_for_plot = policy if args.use_async else None
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(policy=policy),
            subscribers=[
                _saver.VideoSaver(args.out_dir),
                RobotStatePlotter(args.out_dir, broker=broker_for_plot, run_tag=args.run_tag),
            ],
            max_hz=args.fps,
            num_episodes=args.num_episodes,
        )
        print("✅ 运行时初始化成功")
        
        print("\n" + "=" * 60)
        print(f"🎯 任务: {args.prompt}")
        print("=" * 60)
        
        print("\n🏃 开始运行策略...")
        runtime.run()
        environment._robot.enable()
        if args.record_mode and environment:
            print("\n💾 正在保存录制的 episode 数据...")
            environment._save_episode_to_hdf5()
            print("✅ 数据保存完成")
        
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
