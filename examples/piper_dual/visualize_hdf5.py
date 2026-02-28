#!/usr/bin/env python3
"""
可视化PiperDualEnvironment record_mode保存的HDF5文件：
1. 绘制14个关节角度的时序变化曲线图
2. 将图片序列合成为视频并支持预览/保存
3. 支持指定相机（cam_high/cam_left_wrist/cam_right_wrist）
"""
import os
import argparse
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（可选，根据需要调整）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文通用
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文Windows
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class HDF5Visualizer:
    def __init__(self, hdf5_path):
        """初始化可视化器
        Args:
            hdf5_path: HDF5文件路径
        """
        self.hdf5_path = hdf5_path
        self.file = None
        self.qpos_data = None  # 关节数据 (num_steps, 14)
        self.images_data = {}  # 图像数据 {cam_name: (num_steps, 224, 224, 3)}
        self.task = None       # 任务描述
        self.num_steps = 0     # 总步数
        self.fps = 30          # 默认帧率（与采集时一致）
        
        # 关节名称（适配Piper双臂：左臂7个+右臂7个）
        self.joint_names = [
            # 左臂关节
            'Left J1', 'Left J2', 'Left J3', 'Left J4', 'Left J5', 'Left J6', 'Left Gripper',
            # 右臂关节
            'Right J1', 'Right J2', 'Right J3', 'Right J4', 'Right J5', 'Right J6', 'Right Gripper'
        ]
        
        # 加载数据
        self._load_data()

    def _load_data(self):
        """从HDF5文件加载数据"""
        try:
            self.file = h5py.File(self.hdf5_path, 'r')
            print(f"✅ 成功加载HDF5文件: {self.hdf5_path}")
            
            # 加载关节数据
            if 'observations/qpos' in self.file:
                self.qpos_data = self.file['observations/qpos'][:]
                self.num_steps = self.qpos_data.shape[0]
                print(f"📊 关节数据: {self.qpos_data.shape} (步数×关节数)")
            else:
                raise ValueError("HDF5文件中未找到关节数据 (observations/qpos)")
            
            # 加载图像数据
            if 'observations/images' in self.file:
                img_group = self.file['observations/images']
                for cam_name in img_group.keys():
                    self.images_data[cam_name] = img_group[cam_name][:]
                    print(f"🖼️ {cam_name} 图像数据: {self.images_data[cam_name].shape}")
            
            # 加载任务描述
            if 'task' in self.file:
                self.task = self.file['task'][0].decode('utf-8') if isinstance(self.file['task'][0], bytes) else self.file['task'][0]
                print(f"🎯 任务描述: {self.task}")
            
        except Exception as e:
            print(f"❌ 加载HDF5文件失败: {e}")
            raise

    def plot_joint_curves(self, save_path=None):
        """绘制关节角度变化曲线图
        Args:
            save_path: 图片保存路径（None则显示）
        """
        if self.qpos_data is None:
            print("❌ 无关节数据可绘制")
            return
        
        # 创建画布（2行1列，上半部分左臂，下半部分右臂）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Joint Angle Changes\nTask: {self.task}', fontsize=16, fontweight='bold')
        
        # 时间轴（步数）
        time_steps = np.arange(self.num_steps)
        
        # 绘制左臂关节（前7个）
        colors = plt.cm.Set1(np.linspace(0, 1, 7))
        for i in range(7):
            ax1.plot(time_steps, self.qpos_data[:, i], label=self.joint_names[i], color=colors[i], linewidth=1.5)
        ax1.set_title('Left Arm Joints', fontsize=14)
        ax1.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # 绘制右臂关节（后7个）
        colors = plt.cm.Set2(np.linspace(0, 1, 7))
        for i in range(7, 14):
            ax2.plot(time_steps, self.qpos_data[:, i], label=self.joint_names[i], color=colors[i-7], linewidth=1.5)
        ax2.set_title('Right Arm Joints', fontsize=14)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Joint Angle (rad)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 关节曲线图已保存至: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def make_video_from_images(self, cam_name='cam_high', save_path=None, preview=True):
        """将图片序列合成为视频
        Args:
            cam_name: 相机名称（cam_high/cam_left_wrist/cam_right_wrist）
            save_path: 视频保存路径（None则仅预览）
            preview: 是否实时预览
        """
        if cam_name not in self.images_data:
            print(f"❌ 无{cam_name}相机数据，可用相机: {list(self.images_data.keys())}")
            return
        
        images = self.images_data[cam_name]
        # 图像格式为 (num_steps, C, H, W)，需要获取正确的尺寸
        height, width = images.shape[2], images.shape[3]
        
        # 视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 预览窗口设置
        if preview:
            cv2.namedWindow(f'Video Preview: {cam_name}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Video Preview: {cam_name}', 800, 800)
        
        # 视频写入器（如果需要保存）
        video_writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            video_writer = cv2.VideoWriter(save_path, fourcc, self.fps, (width, height))
            print(f"📹 开始生成视频: {save_path}")
        
        # 逐帧处理（RGB→BGR，适配OpenCV）
        for i, frame in enumerate(images):
            # HDF5中保存的是RGB，转换为BGR用于OpenCV
            frame = np.transpose(frame, (1, 2, 0))
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 写入视频
            if video_writer:
                video_writer.write(frame_bgr)
            
            # 预览显示
            if preview:
                cv2.imshow(f'Video Preview: {cam_name}', frame_bgr)
                # 按q退出预览
                if cv2.waitKey(int(1000/self.fps)) & 0xFF == ord('q'):
                    print("⚠️ 预览已终止")
                    break
        
        # 释放资源
        if video_writer:
            video_writer.release()
            print(f"✅ 视频已保存至: {save_path}")
        if preview:
            cv2.destroyAllWindows()

    def close(self):
        """关闭HDF5文件"""
        if self.file:
            self.file.close()

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='可视化Piper双臂HDF5录制文件')
    parser.add_argument('--hdf5_path', required=True, help='HDF5文件路径')
    parser.add_argument('--joint_plot', action='store_true', help='绘制关节变化曲线图')
    parser.add_argument('--joint_save_path', default='output/joint_curves.png', help='关节图保存路径')
    parser.add_argument('--make_video', action='store_true', help='生成视频')
    parser.add_argument('--cam_name', default='cam_high', help='相机名称: cam_high/cam_left_wrist/cam_right_wrist')
    parser.add_argument('--video_save_path', default='output/preview_video.mp4', help='视频保存路径')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--no_preview', action='store_true', help='不预览视频')
    parser.add_argument('--all_cameras', action='store_true', help='保存所有相机视频到output文件夹')
    
    args = parser.parse_args()
    
    # 初始化可视化器
    visualizer = HDF5Visualizer(args.hdf5_path)
    visualizer.fps = args.fps
    
    # 绘制关节曲线图
    if args.joint_plot:
        visualizer.plot_joint_curves(save_path=args.joint_save_path)
    
    # 保存所有相机视频
    if args.all_cameras:
        for cam_name in visualizer.images_data.keys():
            video_path = f'output/{cam_name}_video.mp4'
            visualizer.make_video_from_images(
                cam_name=cam_name,
                save_path=video_path,
                preview=False
            )
    # 生成单个相机视频
    elif args.make_video:
        visualizer.make_video_from_images(
            cam_name=args.cam_name,
            save_path=args.video_save_path,
            preview=not args.no_preview
        )
    
    # 关闭文件
    visualizer.close()
    print("✅ 可视化完成")

if __name__ == '__main__':
    main()