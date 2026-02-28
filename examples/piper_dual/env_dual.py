#!/usr/bin/env python3
"""
双臂 Piper 机械臂环境，用于同时控制和采集双臂数据。
"""
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import time
import threading
import cv2
import h5py
from datetime import datetime
import os

from piper_dual_controller import PiperDualController
from cameras import RealSenseCamera, USBCamera


class PiperDualEnvironment(_environment.Environment):
    """An environment for dual Piper robot arms with multiple cameras (real hardware)."""

    def __init__(
        self,
        left_can_port: str = "can_left",
        right_can_port: str = "can_right",
        camera_fps: int = 30,
        high_camera_id: str = None,
        left_wrist_camera_id: int = None,
        right_wrist_camera_id: int = None,
        max_episode_steps: int = 500,
        initial_left_joint_pos: list = None,
        initial_right_joint_pos: list = None,
        seed: int = 0,
        tele_mode: bool = False,
        prompt: str = "",
        watchdog_timeout: float = 5.0,
        show_usb_camera: bool = False,
        gripper_norm: bool = True,
        action_history_size: int = 30,
        action_horizon: int = 32,
        record_mode: bool = False,
        record_directory: str = "recorded_data",
    ) -> None:
        """Initialize the Piper dual-arm robot environment.
        
        Args:
            left_can_port: CAN bus port name for the left Piper robot (default "can_left").
            right_can_port: CAN bus port name for the right Piper robot (default "can_right").
            camera_fps: Camera frames per second.
            high_camera_id: RealSense camera serial number for cam_high (global view). None for first available.
            left_wrist_camera_id: USB camera device ID for left wrist camera.
            right_wrist_camera_id: USB camera device ID for right wrist camera.
            max_episode_steps: Maximum steps per episode before auto-termination.
            initial_left_joint_pos: Initial left arm joint configuration [j1, j2, j3, j4, j5, j6, gripper].
            initial_right_joint_pos: Initial right arm joint configuration [j1, j2, j3, j4, j5, j6, gripper].
            seed: Random seed (for compatibility; not used in real hardware).
            tele_mode: If True, disables robot motion commands for safe testing.
            prompt: Text prompt to be included in observations.
            show_usb_camera: If True, opens an OpenCV window to preview the USB camera in real time.
            record_mode: If True, records trajectory data during episodes.
            record_directory: Directory to save recorded episodes (default "recorded_data").
        """
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)

        # Threading for parallel hardware polling
        self._obs_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._update_thread = None
        self._update_rate = camera_fps  # Hz
        
        # Watchdog mechanism: detect if background thread is stuck (redundant?)
        self._watchdog_timeout = watchdog_timeout
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
        self._device_healthy = True
        self._robot_enabled = False

        # Initialize dual arm hardware
        self._robot = PiperDualController(
            left_can_port=left_can_port, 
            right_can_port=right_can_port, 
            gripper_norm=gripper_norm
        )
        
        # Initialize cameras: high (global view), left wrist, right wrist (optimized)
        # Note: RealSense D435 supported resolutions: 640x480, 1280x720, 1920x1080
        # Cameras are optional; if None is passed, they are not initialized
        self._high_camera = None
        self._left_wrist_camera = None
        self._right_wrist_camera = None
        
        if high_camera_id is not None:
            self._high_camera = RealSenseCamera(serial_number=high_camera_id, width=640, height=480, fps=camera_fps)
        if left_wrist_camera_id is not None:
            self._left_wrist_camera = USBCamera(camera_id=left_wrist_camera_id, width=224, height=224, fps=camera_fps)
        if right_wrist_camera_id is not None:
            self._right_wrist_camera = USBCamera(camera_id=right_wrist_camera_id, width=224, height=224, fps=camera_fps)
        
        # Episode tracking
        self._done = True
        self._episode_reward = 0.0
        self._step_count = 0
        self._max_episode_steps = max_episode_steps
        
        # Default initial positions for both arms
        if initial_left_joint_pos is None:
            self._initial_left_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self._initial_left_joint_pos = initial_left_joint_pos
            
        if initial_right_joint_pos is None:
            self._initial_right_joint_pos = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            self._initial_right_joint_pos = initial_right_joint_pos

        # Store tele_mode setting (used before _start_hardware to decide whether to enable robot arms)
        self._tele_mode = tele_mode
        
        # Start hardware and background threads
        self._start_hardware()
        print(f"PiperDualEnvironment initialized with CAN ports: Left={left_can_port}, Right={right_can_port}")
        if self._high_camera is not None:
            print(f"  - High camera (RealSense SN {high_camera_id}): {640}x{480}@{camera_fps}fps")
        else:
            print(f"  - High camera: DISABLED")
        if self._left_wrist_camera is not None:
            print(f"  - Left wrist camera (USB ID {left_wrist_camera_id}): {224}x{224}@{camera_fps}fps")
        else:
            print(f"  - Left wrist camera: DISABLED")
        if self._right_wrist_camera is not None:
            print(f"  - Right wrist camera (USB ID {right_wrist_camera_id}): {224}x{224}@{camera_fps}fps")
        else:
            print(f"  - Right wrist camera: DISABLED")

        self._prompt = prompt

        # USB camera preview settings
        self._show_usb_camera = show_usb_camera
        self._usb_window_name = "USB Camera Preview"
        self._usb_window_initialized = show_usb_camera
        if self._high_camera is not None:
            self._display_width = self._high_camera.width // 2
            self._display_height = self._high_camera.height // 2
        else:
            self._display_width = 640
            self._display_height = 360
        self._display_frame_counter = 0
        
        # Action history queue
        self._action_history_size = action_history_size
        self._action_history = []
        self._action_horizon = action_horizon
        
        # Trajectory recording
        self._record_mode = record_mode
        self._record_directory = record_directory
        self._episode_data = []
        self._last_obs = None

    def _start_hardware(self):
        """Starts camera hardware and optionally enables robot arms."""
        # Start cameras (only if configured)
        if self._high_camera is not None:
            self._high_camera.start()
        if self._left_wrist_camera is not None:
            self._left_wrist_camera.start()
        if self._right_wrist_camera is not None:
            self._right_wrist_camera.start()
        
        # 使能机械臂
        try:
            self._robot.enable()
            self._robot_enabled = True
        except Exception as e:
            print(f"Warning during enable: {e}")
            raise RuntimeError(f"Failed to enable robot: {e}")
        
        # Wait a moment for cameras to initialize
        time.sleep(2.0)

    def _update_observation(self):
        """Gathers data from all devices and returns observation."""
        start_ts = time.time()
        
        # 读取全局相机（如果配置了）
        high_frame = None
        high_ts = start_ts
        if self._high_camera is not None:
            try:
                high_frame = self._high_camera.read(timeout=0.5)
                high_ts = time.time()
                if high_frame is None:
                    print(f"⚠️ 警告: 全局相机读取超时或失败")
            except Exception as e:
                print(f"❌ 警告: 全局相机读取异常: {e}")
                high_frame = None
                high_ts = time.time()
        
        # 读取左腕部相机（如果配置了）
        left_wrist_frame = None
        left_wrist_ts = start_ts
        if self._left_wrist_camera is not None:
            try:
                left_wrist_frame = self._left_wrist_camera.read(timeout=0.5)
                left_wrist_ts = time.time()
                if left_wrist_frame is None:
                    print(f"⚠️ 警告: 左腕部相机读取超时或失败")
            except Exception as e:
                print(f"❌ 警告: 左腕部相机读取异常: {e}")
                left_wrist_frame = None
                left_wrist_ts = time.time()
        
        # 读取右腕部相机（如果配置了）
        right_wrist_frame = None
        right_wrist_ts = start_ts
        if self._right_wrist_camera is not None:
            try:
                right_wrist_frame = self._right_wrist_camera.read(timeout=0.5)
                right_wrist_ts = time.time()
                if right_wrist_frame is None:
                    print(f"⚠️ 警告: 右腕部相机读取超时或失败")
            except Exception as e:
                print(f"❌ 警告: 右腕部相机读取异常: {e}")
                right_wrist_frame = None
                right_wrist_ts = time.time()
        
        # 读取双臂机械臂状态 (14维: 左臂7维 + 右臂7维)
        try:
            state = self._robot.get_dual_joint_status()
            robot_ts = time.time()
        except Exception as e:
            print(f"❌ 警告: 机械臂状态读取失败: {e}")
            state = [0.0] * 14
            robot_ts = time.time()
        
        # Process images and build images dict (只处理已配置的相机)
        images_dict = {}
        timestamps_dict = {
            "start": start_ts,
            "robot": robot_ts,
        }
        
        if self._high_camera is not None:
            high_img = self._process_camera_frame(
                high_frame, self._high_camera.width, self._high_camera.height
            )
            images_dict["cam_high"] = high_img
            timestamps_dict["cam_high"] = high_ts
        
        if self._left_wrist_camera is not None:
            left_wrist_img = self._process_camera_frame(
                left_wrist_frame, self._left_wrist_camera.width, self._left_wrist_camera.height
            )
            images_dict["cam_left_wrist"] = left_wrist_img
            timestamps_dict["cam_left_wrist"] = left_wrist_ts
        
        if self._right_wrist_camera is not None:
            right_wrist_img = self._process_camera_frame(
                right_wrist_frame, self._right_wrist_camera.width, self._right_wrist_camera.height
            )
            images_dict["cam_right_wrist"] = right_wrist_img
            timestamps_dict["cam_right_wrist"] = right_wrist_ts

        # Assemble the observation dictionary
        # ZR-0 服务器期望的字段格式：
        # - observation.state
        # - task
        # - 相机图像键名: observation.images.cam_high, observation.images.cam_left_wrist, etc.
        obs = {
            #"observation.state": np.array(state, dtype=np.float32),
            "state": np.array(state, dtype=np.float32),
            "images": images_dict,
            # "task": self._prompt, 
            "prompt": self._prompt,
            #"n_action_steps": self._action_horizon,
        }
        
        # 使用 ZR-0 期望的键名格式: observation.images.xxx
        for cam_name, img in images_dict.items():
            obs["images"][cam_name] = img
            #obs[f"observation.images.{cam_name}"] = img
        # 可选：实时显示 USB 相机画面
        if self._show_usb_camera:
            # self._display_usb_camera_frame(left_wrist_img)
            self._display_usb_camera_frame(right_wrist_img)
        
        return obs

    def _display_usb_camera_frame(self, frame: np.ndarray) -> None:
        """在单独窗口中实时显示 USB 相机画面。"""
        try:
            if frame is None:
                return

            if not self._usb_window_initialized:
                cv2.namedWindow(self._usb_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self._usb_window_name, self._display_width, self._display_height)
                self._usb_window_initialized = True

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            original_height, original_width = frame.shape[:2]
            if original_width == self._display_width and original_height == self._display_height:
                cv2.imshow(self._usb_window_name, frame_bgr)
            else:
                resized_frame = cv2.resize(
                    frame_bgr, 
                    (self._display_width, self._display_height), 
                    interpolation=cv2.INTER_AREA
                )
                cv2.imshow(self._usb_window_name, resized_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("⚠️ 检测到按键 'q'，已关闭 USB 相机实时预览。")
                self._show_usb_camera = False
                cv2.destroyWindow(self._usb_window_name)
                self._usb_window_initialized = False
        except Exception as e:
            print(f"❌ USB 相机预览异常: {e}")
            self._show_usb_camera = False
            try:
                cv2.destroyWindow(self._usb_window_name)
            except Exception:
                pass
            self._usb_window_initialized = False

    def _process_camera_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Processes a raw camera frame (resize, color convert, transpose)."""
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            print("⚠️ 警告: 处理相机帧时输入为 None，已使用黑色图像替代。")
        
        # IMPORTANT: rotate 180 degrees to match ZR-0 train preprocessing
        # frame = np.ascontiguousarray(frame[::-1, ::-1])
        
        target_size = 224
        cur_height, cur_width = frame.shape[:2]
        
        ratio = max(cur_width / target_size, cur_height / target_size)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        
        if ratio > 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
        # 保持BGR格式，与训练数据采集时一致
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pad_h = (target_size - resized_height) // 2
        pad_w = (target_size - resized_width) // 2
        pad_h_remainder = target_size - resized_height - pad_h
        pad_w_remainder = target_size - resized_width - pad_w
        
        padded = np.pad(
            resized_rgb,
            ((pad_h, pad_h_remainder), (pad_w, pad_w_remainder), (0, 0)),
            mode='constant',
            constant_values=0
        )
        # 返回 (H, W, C) 格式，ZR-0 服务端 to_pil_image 期望此格式
        padded_chw = np.transpose(padded, (2, 0, 1))
        # 返回 (C, H, W) 格式，PI-0 服务端 to_pil_image 期望此格式
        return padded_chw
    
    @override
    def reset(self) -> None:
        """Reset the environment: enable robot (if needed), reset counters.
        """
        try:
            if not self._robot_enabled:
                self._robot.enable()
                self._robot_enabled = True
            else:
                print("Robot already enabled")
        except Exception as e:
            print(f"Warning during enable: {e}")
        print("Reset complete. Robot stays at current position.")

        self._done = False
        self._episode_reward = 0.0
        self._step_count = 0
        self._action_history = []
        
        if self._record_mode:
            self._episode_data = []
            self._last_obs = None
        print("Environment reset complete.")

    @override
    def is_episode_complete(self) -> bool:
        """Check if the episode is done."""
        if self._step_count >= self._max_episode_steps:
            self._done = True
        
        return self._done

    @override
    def get_observation(self) -> dict:
        """Return the observation from all sensors."""
        if not self._device_healthy:
            raise RuntimeError("⚠️ 设备处于不健康状态，可能已断开连接。请重启程序。")
        
        start_time = time.time()
        while time.time() - start_time < self._watchdog_timeout:
            obs = self._update_observation()
            if obs is not None:
                if self._record_mode:
                    self._last_obs = obs.copy()
                return obs
            time.sleep(0.0001)
        
        raise RuntimeError(f"⚠️ 看门狗超时：获取观测数据失败（超时阈值: {self._watchdog_timeout}s）。")

    @override
    def apply_action(self, action: dict) -> None:
        """Apply an action to both robot arms.
        
        Expected action format:
        {
            "actions": np.ndarray or list, shape (action_horizon, 14) or (14,)
                       containing [left_j1, ..., left_j6, left_gripper, right_j1, ..., right_j6, right_gripper]
        }
        """
        joint_action = action.get("actions", None)
        if joint_action is None:
            raise ValueError("Action dict must contain 'actions' key with joint positions.")
        
        # ActionChunkBroker已经处理了action chunk，这里直接转换为list
        if isinstance(joint_action, np.ndarray):
            joint_action = joint_action.tolist()
        
        if self._record_mode and self._last_obs is not None:
            step_data = {
                'observation': self._last_obs.copy(),
                'action': np.array(joint_action, dtype=np.float32)
            }
            self._episode_data.append(step_data)
        
        try:
            self._robot.control_dual_joint(joint_action, velocity=100)
        except Exception as e:
            print(f"Error applying action: {e}")
            import traceback
            traceback.print_exc()
            self._done = True
        
        action_np = np.array(joint_action, dtype=np.float32)
        self._action_history.append(action_np)
        if len(self._action_history) > self._action_history_size:
            self._action_history.pop(0)
        
        self._step_count += 1
        reward = 0.0
        self._episode_reward += reward

    # def wait_until_executed(self, action: dict, threshold: float = 0.15, timeout: float = 5.0) -> None:
    #     """Wait until the robot reaches the target position."""
    #     target = action.get("actions")
    #     if target is None:
    #         return
            
    #     target = np.array(target, dtype=np.float32)
    #     if target.ndim > 1:
    #         target = target[0] 
        
    #     target = target.flatten()
    #     if len(target) != 14:
    #         return

    #     start_time = time.time()
    #     while time.time() - start_time < timeout:
    #         try:
    #             current = np.array(self._robot.get_dual_joint_status(), dtype=np.float32)
    #             if len(current) != 14:
    #                 time.sleep(0.01)
    #                 continue
    #             error = np.linalg.norm(target - current)
    #             if error < threshold:
    #                 break
    #             time.sleep(0.01)
    #         except Exception:
    #             break

    def close(self) -> None:
        """Clean up: stop threads, disable robot, and stop cameras."""
        print("🔄 正在关闭 PiperDualEnvironment...")
        
        self._stop_event.set()
        
        if self._update_thread:
            print("⏳ 等待后台线程结束...")
            self._update_thread.join(timeout=3)
            if self._update_thread.is_alive():
                print("⚠️ 警告: 后台线程未在超时时间内结束")

        try:
            print("🔄 正在失能双臂机械臂...")
            if self._robot_enabled:
                #self._robot.disable()
                self._robot_enabled = False
                
                print("✅ 双臂机械臂已失能")
            else:
                print("Robot already disabled")
        except Exception as e:
            print(f"❌ 机械臂失能时出错: {e}")

        if self._high_camera is not None:
            try:
                print("🔄 正在关闭全局相机...")
                self._high_camera.stop()
                print("✅ 全局相机已关闭")
            except Exception as e:
                print(f"❌ 全局相机关闭时出错: {e}")

        if self._left_wrist_camera is not None:
            try:
                print("🔄 正在关闭左腕部相机...")
                self._left_wrist_camera.stop()
                print("✅ 左腕部相机已关闭")
            except Exception as e:
                print(f"❌ 左腕部相机关闭时出错: {e}")
    
        if self._right_wrist_camera is not None:
            try:
                print("🔄 正在关闭右腕部相机...")
                self._right_wrist_camera.stop()
                print("✅ 右腕部相机已关闭")
            except Exception as e:
                print(f"❌ 右腕部相机关闭时出错: {e}")

        if self._usb_window_initialized:
            cv2.destroyWindow(self._usb_window_name)
        
        print("✅ PiperDualEnvironment 已完全关闭")

    def _save_episode_to_hdf5(self):
        """Save the current episode data to an HDF5 file."""
        if not self._episode_data:
            print("No episode data to save.")
            return
        
        os.makedirs(self._record_directory, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prompt = self._prompt if self._prompt else ""
        if prompt:
            safe_prompt = prompt.replace(" ", "_").replace("/", "_").replace("\\", "_")
            safe_prompt = safe_prompt.replace(",", "").replace(".", "")
            max_prompt_len = 80
            if len(safe_prompt) > max_prompt_len:
                safe_prompt = safe_prompt[:max_prompt_len]
            filename = os.path.join(self._record_directory, f"dual_episode_{timestamp_str}_{safe_prompt}.hdf5")
        else:
            filename = os.path.join(self._record_directory, f"dual_episode_{timestamp_str}.hdf5")
        
        print(f"Saving dual-arm episode to {filename}...")
        
        episode_obs_data = []
        for step_data in self._episode_data:
            episode_obs_data.append(step_data['observation'])
        
        if not episode_obs_data:
            print("No observation data to save.")
            return
        
        with h5py.File(filename, "w") as f:
            first_obs = episode_obs_data[0]
            first_action = self._episode_data[0]['action']
            num_steps = len(episode_obs_data)
            
            obs_group = f.create_group("observations")
            img_group = obs_group.create_group("images")
            
            # state_shape = first_obs['observation.state'].shape
            state_shape = first_obs['state'].shape
            # obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['observation.state'].dtype)
            obs_group.create_dataset("qpos", (num_steps,) + state_shape, dtype=first_obs['state'].dtype)
            
            action_shape = first_action.shape
            f.create_dataset("action", (num_steps,) + action_shape, dtype=first_action.dtype)
            
            for cam_name, img in first_obs['images'].items():
                img_shape = img.shape
                img_group.create_dataset(cam_name, (num_steps,) + img_shape, dtype=img.dtype)
                
            # if 'task' in first_obs:
            #     prompt_len = len(first_obs['task'])
            #     fix_prompt_type = np.dtype(f'S{prompt_len}')
            #     f.create_dataset("task", (num_steps,), dtype=fix_prompt_type)
            if 'prompt' in first_obs:
                prompt_len = len(first_obs['prompt'])
                fix_prompt_type = np.dtype(f'S{prompt_len}')
                f.create_dataset("prompt", (num_steps,), dtype=fix_prompt_type)
            
            timestamps_group = None
            if 'timestamps' in first_obs and first_obs['timestamps']:
                timestamps_group = obs_group.create_group("timestamps")
                for ts_key in first_obs['timestamps'].keys():
                    timestamps_group.create_dataset(ts_key, (num_steps,), dtype=np.float64)
            
            for i in range(num_steps):
                # obs_group['qpos'][i] = episode_obs_data[i]['observation.state']
                obs_group['qpos'][i] = episode_obs_data[i]['state']
                f['action'][i] = self._episode_data[i]['action']
                
                for cam_name, img in episode_obs_data[i]['images'].items():
                    img_group[cam_name][i] = img
                # if 'task' in first_obs:
                #     f['task'][i] = episode_obs_data[i]['task']
                if 'prompt' in first_obs:
                    f['prompt'][i] = episode_obs_data[i]['prompt']
                
                if timestamps_group is not None and 'timestamps' in episode_obs_data[i]:
                    for ts_key, ts_value in episode_obs_data[i]['timestamps'].items():
                        if ts_key in timestamps_group:
                            timestamps_group[ts_key][i] = ts_value
                    
        print(f"Successfully saved {num_steps} steps.")
        self._episode_data = []

    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.close()
        except Exception:
            pass
