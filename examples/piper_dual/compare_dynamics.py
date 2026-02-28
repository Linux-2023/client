#!/usr/bin/env python3
"""
对比多次运行的机械臂动力学数据。

由于每次运行生成的 dynamics.png 都是独立的，无法直接在图上对比。
本脚本用于读取 RobotStatePlotter 保存的 .npz 数据文件（或 HDF5 文件），
并在同一张图表上绘制不同实验的对比曲线。

用法:
    python compare_dynamics.py data/piper_dual/dynamics/dynamics_data_1.npz data/piper_dual/dynamics/dynamics_data_2.npz --labels "Exp 1" "Exp 2"
"""

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py

class DynamicsComparator:
    @staticmethod
    def load_data(file_path):
        """加载 .npz 或 .hdf5 文件"""
        file_path = pathlib.Path(file_path)
        data = {}
        
        try:
            if file_path.suffix == '.npz':
                with np.load(file_path) as f:
                    data['timestamps'] = f['timestamps']
                    data['left_joints'] = f['left_joints']
                    data['right_joints'] = f['right_joints']
            elif file_path.suffix == '.hdf5':
                with h5py.File(file_path, 'r') as f:
                    if 'observations/qpos' in f:
                        qpos = f['observations/qpos'][:]
                        # 假设 30Hz
                        data['timestamps'] = np.arange(len(qpos)) / 30.0 
                        dim = qpos.shape[1]
                        data['left_joints'] = qpos[:, :dim//2]
                        data['right_joints'] = qpos[:, dim//2:]
                    else:
                        raise ValueError("HDF5文件中未找到qpos数据")
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
                
            return data
        except Exception as e:
            print(f"❌ 加载文件 {file_path.name} 失败: {e}")
            return None

    @staticmethod
    def calculate_metrics(timestamps, positions):
        """计算速度、加速度及其统计指标（如L2范数）"""
        
        # 预处理：时间对齐
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        if len(timestamps) > 0:
            timestamps = timestamps - timestamps[0]
            
        # 移除夹爪 (如果 > 6维度)
        if positions.shape[1] > 6:
            positions = positions[:, :-1]
            
        # 计算 dt
        dt = np.diff(timestamps)
        dt = np.maximum(dt, 1e-4)[:, np.newaxis]
        
        # 速度
        velocities = np.diff(positions, axis=0) / dt
        t_vel = timestamps[1:]
        
        # 简单过滤第一帧异常
        if len(velocities) > 1:
             velocities[0] = 0

        # 加速度
        accelerations = np.diff(velocities, axis=0) / dt[:-1]
        t_acc = timestamps[2:]
        if len(accelerations) > 1:
            accelerations[0] = 0
            
        # 计算 L2 范数 (代表整体运动强度)
        vel_norm = np.linalg.norm(velocities, axis=1)
        acc_norm = np.linalg.norm(accelerations, axis=1)
        
        # 计算 Max Abs (代表最大单关节负荷)
        vel_max = np.max(np.abs(velocities), axis=1)
        acc_max = np.max(np.abs(accelerations), axis=1)

        return {
            't_vel': t_vel, 'vel_norm': vel_norm, 'vel_max': vel_max, 'velocities': velocities,
            't_acc': t_acc, 'acc_norm': acc_norm, 'acc_max': acc_max, 'accelerations': accelerations
        }

    def plot_comparison(self, file_paths, labels=None, save_path="comparison_dynamics.png"):
        if not labels:
            labels = [p.stem for p in map(pathlib.Path, file_paths)]
        
        if len(file_paths) != len(labels):
            print("⚠️ 标签数量与文件数量不匹配，将使用默认标签")
            labels = [p.stem for p in map(pathlib.Path, file_paths)]

        loaded_datasets = []
        for fp in file_paths:
            d = self.load_data(fp)
            if d: loaded_datasets.append(d)
        
        if not loaded_datasets:
            print("❌ 没有有效数据可绘制")
            return

        # 创建图表
        # 2x2: 左臂速度对比，右臂速度对比，左臂加速度对比，右臂加速度对比
        # 这里我们对比 "平均速度(L2 Norm)" 以简化图表，避免线条过多
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Multi-Experiment Dynamics Comparison (Metric: L2 Norm)', fontsize=16)
        
        metrics = []
        for data in loaded_datasets:
             metrics.append({
                 'left': self.calculate_metrics(data['timestamps'], data['left_joints']),
                 'right': self.calculate_metrics(data['timestamps'], data['right_joints'])
             })

        # 绘图函数
        def plot_metric(ax, metric_key, title, ylabel):
            for i, m in enumerate(metrics):
                l_m = m['left']
                r_m = m['right']
                
                # 选择数据源
                if 'Left' in title:
                    t = l_m['t_vel'] if 'vel' in metric_key else l_m['t_acc']
                    y = l_m[metric_key]
                else:
                    t = r_m['t_vel'] if 'vel' in metric_key else r_m['t_acc']
                    y = r_m[metric_key]
                
                # 平滑处理（可选，为了让对比更清晰）
                # 这里不做平滑，保持真实
                ax.plot(t, y, label=labels[i], alpha=0.8, linewidth=1.5)
            
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 1. 左臂速度对比
        plot_metric(axs[0, 0], 'vel_norm', 'Left Arm Velocity (L2 Norm)', 'Speed (rad/s)')
        
        # 2. 右臂速度对比
        plot_metric(axs[0, 1], 'vel_norm', 'Right Arm Velocity (L2 Norm)', 'Speed (rad/s)')
        
        # 3. 左臂加速度对比
        plot_metric(axs[1, 0], 'acc_norm', 'Left Arm Acceleration (L2 Norm)', 'Accel (rad/s²)')
        axs[1, 0].set_xlabel('Time (s)')
        
        # 4. 右臂加速度对比
        plot_metric(axs[1, 1], 'acc_norm', 'Right Arm Acceleration (L2 Norm)', 'Accel (rad/s²)')
        axs[1, 1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ 对比图已保存至: {save_path}")
        
        # 额外生成一张特定关节 (J1, J4) 的对比图
        self.plot_joint_comparison(metrics, labels, save_path.replace(".png", "_joints.png"))

    def plot_joint_comparison(self, metrics, labels, save_path):
        """对比特定关节 (如 J1 底座, J4 肘部)"""
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Single Joint Dynamics Comparison (Left Arm)', fontsize=16)

        # 关注的关节索引 (0-based)
        joints_to_compare = [0, 3] # J1, J4
        joint_names = ["J1 (Base)", "J4 (Elbow)"]
        
        for i, j_idx in enumerate(joints_to_compare):
            # 速度
            ax_vel = axs[0, i]
            for k, m in enumerate(metrics):
                t = m['left']['t_vel']
                vel = m['left']['velocities'][:, j_idx]
                ax_vel.plot(t, vel, label=labels[k])
            ax_vel.set_title(f'Left {joint_names[i]} Velocity')
            ax_vel.set_ylabel('rad/s')
            ax_vel.grid(True)
            ax_vel.legend()

            # 加速度
            ax_acc = axs[1, i]
            for k, m in enumerate(metrics):
                t = m['left']['t_acc']
                acc = m['left']['accelerations'][:, j_idx]
                ax_acc.plot(t, acc, label=labels[k])
            ax_acc.set_title(f'Left {joint_names[i]} Acceleration')
            ax_acc.set_ylabel('rad/s²')
            ax_acc.set_xlabel('Time (s)')
            ax_acc.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✅ 关节对比图已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比多个实验的动力学数据")
    parser.add_argument("files", nargs="+", help=".npz 或 .hdf5 数据文件路径")
    parser.add_argument("--labels", nargs="+", help="对应实验的图例标签", default=[])
    parser.add_argument("--out", default="comparison_result.png", help="输出图片路径")
    
    args = parser.parse_args()
    
    comparator = DynamicsComparator()
    comparator.plot_comparison(args.files, args.labels, args.out)
