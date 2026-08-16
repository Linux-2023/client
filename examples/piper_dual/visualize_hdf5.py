#!/usr/bin/env python3
"""可视化 Piper 双臂 HDF5 录制文件；新 schema 会转调到自包含渲染器。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from pathlib import Path
import warnings

import cv2
import h5py
import numpy as np

from render_dataset import LEGACY_SCHEMA_VERSION
from render_dataset import SCHEMA_VERSION
from render_dataset import detect_schema
from render_dataset import render_episode


warnings.filterwarnings('ignore')


class HDF5Visualizer:
    def __init__(self, hdf5_path):
        """初始化可视化器。"""
        self.hdf5_path = hdf5_path
        self.file = None
        self.qpos_data = None
        self.images_data = {}
        self.task = None
        self.num_steps = 0
        self.fps = 30
        self.joint_names = [
            'Left J1', 'Left J2', 'Left J3', 'Left J4', 'Left J5', 'Left J6', 'Left Gripper',
            'Right J1', 'Right J2', 'Right J3', 'Right J4', 'Right J5', 'Right J6', 'Right Gripper',
        ]
        self._load_data()

    def _load_data(self):
        """从 HDF5 文件加载数据。"""
        try:
            self.file = h5py.File(self.hdf5_path, 'r')
            print(f"✅ 成功加载 HDF5 文件: {self.hdf5_path}")

            if 'observations/qpos' in self.file:
                self.qpos_data = self.file['observations/qpos'][:]
                self.num_steps = self.qpos_data.shape[0]
                print(f"📊 关节数据: {self.qpos_data.shape} (步数×关节数)")
            else:
                raise ValueError('HDF5 文件中未找到关节数据 (observations/qpos)')

            if 'observations/images' in self.file:
                img_group = self.file['observations/images']
                for cam_name in img_group.keys():
                    self.images_data[cam_name] = img_group[cam_name][:]
                    print(f"🖼️ {cam_name} 图像数据: {self.images_data[cam_name].shape}")

            if 'task' in self.file:
                first = self.file['task'][0]
                self.task = first.decode('utf-8') if isinstance(first, bytes) else first
                print(f"🎯 任务描述: {self.task}")
        except Exception as exc:
            print(f"❌ 加载 HDF5 文件失败: {exc}")
            raise

    def plot_joint_curves(self, save_path=None):
        """绘制关节角度变化曲线图。"""
        if self.qpos_data is None:
            print('❌ 无关节数据可绘制')
            return

        plt = _load_matplotlib()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Joint Angle Changes\nTask: {self.task}', fontsize=16, fontweight='bold')

        time_steps = np.arange(self.num_steps)

        colors = plt.cm.Set1(np.linspace(0, 1, 7))
        for i in range(7):
            ax1.plot(time_steps, self.qpos_data[:, i], label=self.joint_names[i], color=colors[i], linewidth=1.5)
        ax1.set_title('Left Arm Joints', fontsize=14)
        ax1.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        colors = plt.cm.Set2(np.linspace(0, 1, 7))
        for i in range(7, 14):
            ax2.plot(time_steps, self.qpos_data[:, i], label=self.joint_names[i], color=colors[i - 7], linewidth=1.5)
        ax2.set_title('Right Arm Joints', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 关节曲线图已保存至: {save_path}")
        else:
            plt.show()
        plt.close()

    def make_video_from_images(self, cam_name='cam_high', save_path=None, preview=True):
        """将图片序列合成为视频。"""
        if cam_name not in self.images_data:
            print(f"❌ 无 {cam_name} 相机数据，可用相机: {list(self.images_data.keys())}")
            return

        images = self.images_data[cam_name]
        height, width = images.shape[2], images.shape[3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if preview:
            cv2.namedWindow(f'Video Preview: {cam_name}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Video Preview: {cam_name}', 800, 800)

        video_writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (width, height))
            print(f"📹 开始生成视频: {save_path}")

        for frame in images:
            frame = np.transpose(frame, (1, 2, 0))
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if video_writer:
                video_writer.write(frame_bgr)
            if preview:
                cv2.imshow(f'Video Preview: {cam_name}', frame_bgr)
                if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                    print('⚠️ 预览已终止')
                    break

        if video_writer:
            video_writer.release()
            print(f"✅ 视频已保存至: {save_path}")
        if preview:
            cv2.destroyAllWindows()

    def close(self):
        """关闭 HDF5 文件。"""
        if self.file:
            self.file.close()


def _is_new_schema_episode(hdf5_path: str | Path) -> bool:
    return detect_schema(Path(hdf5_path)) == SCHEMA_VERSION



def _default_render_output_dir(hdf5_path: str | Path) -> Path:
    path = Path(hdf5_path)
    return path.parent / f'{path.stem}_rendered'


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f'matplotlib is required for joint plotting: {exc}') from exc
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return plt


def main():
    parser = argparse.ArgumentParser(description='可视化 Piper 双臂 HDF5 录制文件')
    parser.add_argument('--hdf5_path', required=True, help='HDF5 文件路径')
    parser.add_argument('--joint_plot', action='store_true', help='绘制关节变化曲线图')
    parser.add_argument('--joint_save_path', default='output/joint_curves.png', help='关节图保存路径')
    parser.add_argument('--make_video', action='store_true', help='生成视频')
    parser.add_argument('--cam_name', default='cam_high', help='相机名称: cam_high/cam_left_wrist/cam_right_wrist')
    parser.add_argument('--video_save_path', default='output/preview_video.mp4', help='视频保存路径')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--no_preview', action='store_true', help='不预览视频')
    parser.add_argument('--all_cameras', action='store_true', help='保存所有相机视频到 output 文件夹')
    parser.add_argument('--output_dir', default=None, help='新 schema 渲染输出目录（默认: <hdf5>_rendered）')
    parser.add_argument('--no_plots', action='store_true', help='新 schema 渲染时跳过 state_action.png')

    args = parser.parse_args()

    schema = detect_schema(Path(args.hdf5_path))
    if schema == SCHEMA_VERSION:
        output_dir = Path(args.output_dir) if args.output_dir else _default_render_output_dir(args.hdf5_path)
        report = render_episode(Path(args.hdf5_path), output_dir, fps=args.fps, make_plots=not args.no_plots)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if schema != LEGACY_SCHEMA_VERSION:
        raise ValueError(f'unsupported HDF5 schema for {args.hdf5_path}: {schema!r}')

    visualizer = HDF5Visualizer(args.hdf5_path)
    visualizer.fps = args.fps

    if args.joint_plot:
        visualizer.plot_joint_curves(save_path=args.joint_save_path)

    if args.all_cameras:
        for cam_name in visualizer.images_data.keys():
            video_path = f'output/{cam_name}_video.mp4'
            visualizer.make_video_from_images(
                cam_name=cam_name,
                save_path=video_path,
                preview=False,
            )
    elif args.make_video:
        visualizer.make_video_from_images(
            cam_name=args.cam_name,
            save_path=args.video_save_path,
            preview=not args.no_preview,
        )

    visualizer.close()
    print('✅ 可视化完成')


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
