#!/usr/bin/env python3
"""
可视化机械臂的速度和加速度变化。

本脚本提供两种使用方式：
1. **离线分析模式（推荐，无需修改原有代码）**：
   直接运行本脚本，分析由 --record_mode 生成的 HDF5 文件。
   示例：python plot_dynamics.py recorded_data/your_episode.hdf5

2. **在线集成模式（需要微小修改 main_dual.py）**：
   作为 Subscriber 集成到 main_dual.py 中，在程序结束时自动生成图表。
"""

import argparse
import csv
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# 尝试导入 Subscriber 基类，如果失败（比如在没有 openpi_client 环境下运行离线分析），则忽略
try:
    from openpi_client.runtime import subscriber as _subscriber
    HAS_OPENPI = True
except ImportError:
    HAS_OPENPI = False
    _subscriber = object  # 占位符

class DynamicsFunction:
    """处理物理量计算的核心逻辑"""
    
    @staticmethod
    def calculate_dynamics(timestamps, positions):
        """
        计算速度和加速度
        Args:
            timestamps: 时间戳数组 (N,) 单位: 秒
            positions: 位置数组 (N, D) 单位: 弧度 (rad)
        Returns:
            velocities: 速度 (N-1, D) 单位: rad/s
            accelerations: 加速度 (N-2, D) 单位: rad/s²
        """
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        
        # 长度对齐
        min_len = min(len(timestamps), len(positions))
        timestamps = timestamps[:min_len]
        positions = positions[:min_len]
        
        if len(timestamps) < 5: # 需要更多点来进行平滑
            return None, None, None, None

        # --- 优化核心：使用平滑的 dt ---
        # 原始 dt 含有系统调度抖动，直接用于分母会导致加速度巨大且不真实。
        # 我们计算 dt 的中位数，代表控制循环的真实周期。
        raw_dt = np.diff(timestamps)
        median_dt = np.median(raw_dt)
        
        # 如果 dt 异常（比如数据中断），则保留原始 dt；否则使用稳定的 dt
        # 允许 20% 的 jitter，超过这个范围可能真的是丢帧，需要用 raw_dt
        dt_jitter = np.abs(raw_dt - median_dt)
        smooth_dt = np.where(dt_jitter < 0.2 * median_dt, median_dt, raw_dt)
        
        # 防止 dt 过小导致除零爆炸 (限制最小 1ms)
        dt = np.maximum(smooth_dt, 0.001)[:, np.newaxis]
        
        # --- 1. 计算速度 (rad/s) ---
        velocities = np.diff(positions, axis=0) / dt
        t_vel = timestamps[1:]
        
        # 过滤第一帧的速度跳变 (初始化时的瞬移)
        if len(velocities) > 1:
            # 简单的阈值：如果第一帧速度 > 后续平均速度的10倍 + 1.0 rad/s
            mean_vel_mag = np.mean(np.abs(velocities[5:]), axis=0) # 跳过前几帧计算均值
            is_outlier = np.abs(velocities[0]) > (mean_vel_mag * 10 + 2.0)
            velocities[0] = np.where(is_outlier, 0, velocities[0])

        # --- 2. 计算加速度 (rad/s²) ---
        # 再次对速度差分。这里 dt 需要对应 velocity 的时间间隔
        dt_acc = dt[:-1]
        accelerations = np.diff(velocities, axis=0) / dt_acc
        t_acc = timestamps[2:]
        
        # 过滤前几帧的加速度震荡
        if len(accelerations) > 2:
            accelerations[:2] = 0
            
        return velocities, accelerations, t_vel, t_acc

    @staticmethod
    def detect_chunk_jumps(delta_cmd, rel_t_delta, action_horizon=15, chunk_boundaries=None):
        """
        确定 Chunk 边界并量化跳变。

        支持两种边界模式：
        - chunk_boundaries 为 None：按固定周期 action_horizon 推断边界（兼容同步模式）
        - chunk_boundaries 为步数列表（来自 ActionChunkBroker_RTC._chunk_boundaries）：
          直接使用实际边界，适用于异步模式（真实对齐）。

        Returns:
            boundary_indices : np.ndarray  在 delta_cmd 坐标系下的索引
            jump_stats       : list of dict  每个边界的统计信息
        """
        n = len(delta_cmd)
        jump_l2 = np.linalg.norm(delta_cmd, axis=1)   # (N-1,)

        if chunk_boundaries is not None and len(chunk_boundaries) > 0:
            # 实际边界模式：step s 对应 delta[s-1] = cmd[s] - cmd[s-1]
            bi_list = [s - 1 for s in chunk_boundaries if 0 < s <= n]
            boundary_indices = np.array(bi_list, dtype=int)
            boundary_indices = boundary_indices[boundary_indices < n]
            mode_label = f"实际边界 ({len(boundary_indices)} 次)"
        else:
            # 固定周期边界（兼容老逻辑）：delta[action_horizon-1], delta[2*action_horizon-1], ...
            boundary_indices = np.arange(action_horizon - 1, n, action_horizon)
            mode_label = f"固定周期 action_horizon={action_horizon}"

        if len(boundary_indices) == 0:
            return np.array([], dtype=int), []

        jump_stats = []
        for bi in boundary_indices:
            d = delta_cmd[bi]                      # (D,)
            jump_stats.append({
                'step'     : int(bi + 1),           # cmd step index (bi+1 对应 delta[bi])
                'l2'       : float(jump_l2[bi]),
                'max_joint': float(np.max(np.abs(d))),
                'per_joint': np.abs(d).tolist(),
            })

        # --- 诊断：打印 L2 最大的前10步，验证是否在 chunk 边界 ---
        top_k = min(10, n)
        top_idx = np.argsort(jump_l2)[::-1][:top_k]
        print(f"\n{'─'*65}")
        print(f"  [诊断] delta_cmd 中 L2 最大的前{top_k}步 ({mode_label})")
        print(f"  {'delta索引':>8}  {'step_idx':>8}  {'step%horizon':>12}  {'偏离边界':>8}  {'L2':>8}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*8}")
        boundary_steps_set = set(int(bi + 1) for bi in boundary_indices)
        for rank, di in enumerate(top_idx):
            step_idx = di + 1
            if chunk_boundaries is not None:
                on_boundary = "✓ 边界" if step_idx in boundary_steps_set else "  非边界"
                offset_str = f"{step_idx}"
            else:
                offset = (step_idx) % action_horizon
                on_boundary = "✓ 边界" if offset == 0 else f"  内部+{offset}"
                offset_str = f"{offset}"
            print(f"  {di:>8}  {step_idx:>8}  {offset_str:>12}  {on_boundary:>8}  {jump_l2[di]:>8.4f}")
        print(f"{'─'*65}\n")

        return boundary_indices, jump_stats

    @staticmethod
    def plot_continuity(timestamps, cmd_positions, measured_positions, save_path, arm_name="Left", action_horizon=15, chunk_boundaries=None, csv_path=None, run_tag=""):
        """
        可视化动作连续性：对比指令位置与其差分（指令速度/跳变）
        """
        timestamps = np.array(timestamps)
        cmd_pos = np.array(cmd_positions)
        meas_pos = np.array(measured_positions)
        
        # 对齐长度
        min_len = min(len(timestamps), len(cmd_pos), len(meas_pos))
        timestamps = timestamps[:min_len]
        cmd_pos = cmd_pos[:min_len]
        meas_pos = meas_pos[:min_len]
        
        if min_len < 2:
            return

        # 同样移除夹爪
        if cmd_pos.shape[1] > 6: 
            cmd_pos = cmd_pos[:, :-1]
            meas_pos = meas_pos[:, :-1]

        # 1. 计算指令的差分 (即相邻帧的指令跳变)
        delta_cmd = np.diff(cmd_pos, axis=0) # (N-1, D)

        # 横轴使用步数（step index），避免同步推理时时间不均匀带来的误导
        step_axis       = np.arange(min_len)          # (N,)   for cmd/meas
        step_axis_delta = np.arange(1, min_len)       # (N-1,) for delta (delta[i]=cmd[i+1]-cmd[i])

        # --- 按固定周期或实际边界定位 Chunk 跳变 ---
        boundary_indices, jump_stats = DynamicsFunction.detect_chunk_jumps(
            delta_cmd, step_axis_delta, action_horizon=action_horizon, chunk_boundaries=chunk_boundaries
        )
        # 边界在步数坐标下直接就是 step_axis_delta[boundary_indices]
        boundary_steps = step_axis_delta[boundary_indices] if len(boundary_indices) > 0 else []

        # --- 展示每个关节的 Measured vs Command ---
        num_joints = cmd_pos.shape[1] 
        # 布局：前 num_joints 个子图画每个关节的 Position Tracking，最后两行画 Delta 和 Velocity
        total_rows = num_joints + 2
        
        # 调整画布大小，防止太挤
        fig, axs = plt.subplots(total_rows, 1, figsize=(15, 3 * total_rows), sharex=True)
        fig.suptitle(f'{arm_name} Arm Detail Analysis (Position Tracking per Joint)', fontsize=16)

        # 1. 绘制每个关节的位置跟踪，chunk 边界加竖线
        for j in range(num_joints):
            ax = axs[j]
            ax.plot(step_axis, meas_pos[:, j], label='Measured', color='gray', alpha=0.5, linewidth=2)
            ax.plot(step_axis, cmd_pos[:, j], label='Command', color='blue', linestyle='--', linewidth=1)
            for k, bs in enumerate(boundary_steps):
                ax.axvline(bs, color='red', linewidth=0.8, linestyle=':', alpha=0.6,
                           label='Chunk boundary' if (j == 0 and k == 0) else '')
            ax.set_title(f'Joint {j+1}')
            ax.set_ylabel('Rad')
            ax.grid(True, alpha=0.3)
            
        # 统一在第一个关节图显示图例
        axs[0].legend(loc='upper right')

        # 2. Command Jumps (All Joints)
        delta_cmd_ax = axs[num_joints]
        for i in range(delta_cmd.shape[1]):
            delta_cmd_ax.plot(step_axis_delta, delta_cmd[:, i], label=f'J{i+1}', alpha=0.7)
        # 在 delta 图上用散点标出 chunk 边界
        if len(boundary_indices) > 0:
            jump_l2 = np.linalg.norm(delta_cmd, axis=1)
            delta_cmd_ax.scatter(
                step_axis_delta[boundary_indices],
                jump_l2[boundary_indices],
                color='red', zorder=5, s=60, marker='v', label='Chunk jump'
            )
            for k, (bs, js) in enumerate(zip(boundary_steps, jump_stats)):
                delta_cmd_ax.axvline(bs, color='red', linewidth=1.0, linestyle='--', alpha=0.5)
                delta_cmd_ax.annotate(
                    f"#{k+1}\nL2={js['l2']:.3f}",
                    xy=(bs, js['l2']), xytext=(bs + 2, js['l2'] * 1.05),
                    fontsize=7, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                )
        delta_cmd_ax.set_title('Command Step Changes (Delta Action)  —  red markers = Chunk boundaries')
        delta_cmd_ax.set_ylabel('Delta (rad)')
        delta_cmd_ax.legend(ncol=min(7, num_joints + 1), fontsize='small', loc='upper right')
        delta_cmd_ax.grid(True, alpha=0.3)
        
        # 3. Command Velocity (Δrad/step，不除以实际 dt 避免抖动放大)
        vel_ax = axs[num_joints + 1]
        # 直接用差分值作为每步速度（单位 rad/step，物理意义清晰）
        for i in range(delta_cmd.shape[1]):
            vel_ax.plot(step_axis_delta, delta_cmd[:, i], alpha=0.6)
        for bs in boundary_steps:
            vel_ax.axvline(bs, color='red', linewidth=0.8, linestyle='--', alpha=0.5)
        vel_ax.set_title('Command Delta per Step (rad/step)')
        vel_ax.set_ylabel('Δrad/step')
        vel_ax.set_xlabel('Step')
        vel_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95) # 留出标题空间
        plt.savefig(save_path)
        plt.close(fig)
        print(f"✅ 详细连续性分析图({arm_name})已保存至: {save_path}")

        # --- 量化报告 (简化为单行摘要，详细数据写入 CSV) ---
        if len(jump_stats) == 0:
            print(f"[{arm_name}] ℹ️  未检测到显著 Chunk 跳变。")
        else:
            all_l2  = [s['l2']        for s in jump_stats]
            all_max = [s['max_joint'] for s in jump_stats]
            print(f"[{arm_name}] Chunk 跳变: {len(jump_stats)} 次  "
                  f"均值 L2={np.mean(all_l2):.4f} rad  "
                  f"最大 L2={np.max(all_l2):.4f} rad  "
                  f"MaxJoint={np.max(all_max):.4f} rad")

            # 追加写入 CSV（每运行一次，追加当次的所有边界数据）
            if csv_path is not None:
                DynamicsFunction.save_jump_stats_to_csv(
                    jump_stats, arm_name=arm_name, csv_path=csv_path, run_tag=run_tag
                )


    @staticmethod
    def save_jump_stats_to_csv(jump_stats, arm_name, csv_path, run_tag=""):
        """
        追加写入 CSV。每次运行调用一次（左臂 / 右臂各一次），
        在文件末尾追加当次全部 chunk 边界的数据行。

        CSV 列: run_time, run_tag, arm, chunk_idx, boundary_step, L2, MaxJoint, J1..J6
        """
        csv_path = pathlib.Path(csv_path)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        # 动态确定关节数
        num_joints = len(jump_stats[0]['per_joint']) if jump_stats else 6
        joint_cols = [f"J{i+1}" for i in range(num_joints)]

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["run_time", "run_tag", "arm", "chunk_idx",
                     "boundary_step", "L2", "MaxJoint"] + joint_cols
                )
            run_time = time.strftime("%Y-%m-%d %H:%M:%S")
            for k, s in enumerate(jump_stats):
                row = [
                    run_time, run_tag, arm_name, k + 1,
                    s["step"], f"{s['l2']:.6f}", f"{s['max_joint']:.6f}",
                ] + [f"{v:.6f}" for v in s["per_joint"]]
                writer.writerow(row)
        print(f"💾 Chunk 跳变数据已追加至: {csv_path}  ({len(jump_stats)} 行, arm={arm_name})")


    @staticmethod
    def plot_and_save(timestamps, left_pos, right_pos, save_path, title_suffix="", left_cmd=None, right_cmd=None, action_horizon=15, chunk_boundaries=None, csv_path=None, run_tag=""):
        """绘制并保存图表"""
        # 1. 基础动力学 (Measured)
        l_vel, l_acc, t_l_vel, t_l_acc = DynamicsFunction.calculate_dynamics(timestamps, left_pos)
        r_vel, r_acc, t_r_vel, t_r_acc = DynamicsFunction.calculate_dynamics(timestamps, right_pos)
        
        if l_vel is None:
            print("⚠️ 数据点太少，无法绘制")
            return
            
        # 智能判断是否移除夹爪
        if l_vel.shape[1] == 7:
            l_vel = l_vel[:, :-1]
            l_acc = l_acc[:, :-1]
        if r_vel.shape[1] == 7:
            r_vel = r_vel[:, :-1]
            r_acc = r_acc[:, :-1]

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Robot Arm Dynamics Analysis {title_suffix}\n(Smoothed dt for Physical Consistency)', fontsize=14)

        # 辅助绘图函数
        def plot_data(ax, time, data, title, y_label):
            rel_time = time - timestamps[0]
            for i in range(data.shape[1]):
                ax.plot(rel_time, data[:, i], label=f'J{i+1}', alpha=0.8, linewidth=1)
            ax.set_title(title)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize='x-small', ncol=2, loc='upper right')

        plot_data(axs[0, 0], t_l_vel, l_vel, 'Left Arm Velocities (Measured)', 'Vel (rad/s)')
        plot_data(axs[0, 1], t_r_vel, r_vel, 'Right Arm Velocities (Measured)', 'Vel (rad/s)')
        
        plot_data(axs[1, 0], t_l_acc, l_acc, 'Left Arm Accelerations (Measured)', 'Acc (rad/s²)')
        axs[1, 0].set_xlabel('Time (s)')
        
        plot_data(axs[1, 1], t_r_acc, r_acc, 'Right Arm Accelerations (Measured)', 'Acc (rad/s²)')
        axs[1, 1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"✅ 动力学图表已保存至: {save_path}")
        
        # 2. 连续性分析 (如果提供了Action数据)
        if left_cmd is not None and len(left_cmd) > 0:
            cont_path_left = str(save_path).replace(".png", "_continuity_left.png")
            DynamicsFunction.plot_continuity(timestamps, left_cmd, left_pos, cont_path_left, "Left", action_horizon=action_horizon, chunk_boundaries=chunk_boundaries, csv_path=csv_path, run_tag=run_tag)
            
        if right_cmd is not None and len(right_cmd) > 0:
            cont_path_right = str(save_path).replace(".png", "_continuity_right.png")
            DynamicsFunction.plot_continuity(timestamps, right_cmd, right_pos, cont_path_right, "Right", action_horizon=action_horizon, chunk_boundaries=chunk_boundaries, csv_path=csv_path, run_tag=run_tag)


class RobotStatePlotter(_subscriber.Subscriber):
    """
    可集成到 main_dual.py 的 Subscriber。
    在运行时收集数据，结束后绘图。
    """
    def __init__(self, save_dir: pathlib.Path = pathlib.Path("data/piper_dual/dynamics"), broker=None, run_tag: str = ""):
        self.save_dir = pathlib.Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamps = []
        self.left_joints = []
        self.right_joints = []
        self.cmd_left_joints = []
        self.cmd_right_joints = []
        self._start_time = None
        # 可选：ActionChunkBroker_RTC 实例，用于获取真实 chunk 边界
        self._broker = broker
        # 用户指定的实验标签（用于 CSV run_tag 列），为空时用时间戳
        self._run_tag = run_tag

    def on_episode_start(self) -> None:
        self.timestamps = []
        self.left_joints = []
        self.right_joints = []
        self.cmd_left_joints = []
        self.cmd_right_joints = []
        self._start_time = time.time()

    def on_step(self, observation: dict, action: dict) -> None:
        # 记录时间
        self.timestamps.append(time.time())
        
        # 1. 提取 Measured Data (observation)
        obs = observation
        qpos = None
        if "qpos" in obs:
            qpos = obs["qpos"]
        elif "state" in obs:
            qpos = obs["state"]
            
        if qpos is not None:
            qpos = np.array(qpos)
            mid = len(qpos) // 2
            self.left_joints.append(qpos[:mid])
            self.right_joints.append(qpos[mid:])
            
        # 2. 提取 Command Data (action)
        # action={'actions': np.array([...])} 通常包含左右臂
        cmd_actions = action.get("actions")
        if cmd_actions is not None:
            cmd_actions = np.array(cmd_actions)
            # 假设 action 也是左右各一半
            # 同样需要确保 cmd_actions 长度与 qpos 一致 (通常是14或12)
            c_mid = len(cmd_actions) // 2
            self.cmd_left_joints.append(cmd_actions[:c_mid])
            self.cmd_right_joints.append(cmd_actions[c_mid:])

    def on_episode_end(self) -> None:
        if not self.timestamps:
            return
            
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. 保存原始数据 (以便后续对比)
        data_path = self.save_dir / f"dynamics_data_{timestamp_str}.npz"
        np.savez(data_path, 
                 timestamps=np.array(self.timestamps),
                 left_joints=np.array(self.left_joints),
                 right_joints=np.array(self.right_joints),
                 cmd_left_joints=np.array(self.cmd_left_joints) if self.cmd_left_joints else np.array([]),
                 cmd_right_joints=np.array(self.cmd_right_joints) if self.cmd_right_joints else np.array([]))
        print(f"💾 原始动力学数据已保存至: {data_path}")

        # 从 broker 获取真实 chunk 边界（异步模式），否则 plot_and_save 使用固定周期
        chunk_boundaries = None
        if self._broker is not None and hasattr(self._broker, '_chunk_boundaries'):
            chunk_boundaries = list(self._broker._chunk_boundaries)
            print(f"[RobotStatePlotter] 使用实际 chunk 边界: {len(chunk_boundaries)} 次")

        # run_tag: 用户指定则使用，否则 fallback 至时间戳
        effective_tag = self._run_tag if self._run_tag else timestamp_str

        # 2. 绘制图表
        save_path = self.save_dir / f"dynamics_{timestamp_str}.png"
        # 持久化 CSV 路径（放在 save_dir 的父目录，跨 episode 累计）
        csv_path = self.save_dir.parent / "chunk_jump_stats.csv"
        print("\n[RobotStatePlotter] 正在生成动力学图表...")
        DynamicsFunction.plot_and_save(
            self.timestamps, 
            self.left_joints, 
            self.right_joints, 
            save_path,
            title_suffix="(Runtime Recording)",
            left_cmd=self.cmd_left_joints,
            right_cmd=self.cmd_right_joints,
            chunk_boundaries=chunk_boundaries,
            csv_path=csv_path,
            run_tag=effective_tag,
        )


def analyze_hdf5(file_path, fps=30, action_horizon=15):
    """离线分析 HDF5 文件"""
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"📂 正在分析文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 读取 qpos
            if 'observations/qpos' in f:
                qpos = f['observations/qpos'][:]
            elif 'qpos' in f: # 兼容其他格式
                qpos = f['qpos'][:]
            else:
                print("❌ HDF5 文件中未找到 observations/qpos")
                return
            
            # 尝试读取 action
            actions = None
            if 'action' in f:
                actions = f['action'][:]
            elif 'actions' in f:
                actions = f['actions'][:]
                
            # 尝试推断时间
            num_steps = len(qpos)
            if 'fps' in f.attrs:
                fps = f.attrs['fps']
                
            timestamps = np.arange(num_steps) / fps
            
            # 左右臂分割
            dim = qpos.shape[1]
            left_qpos = qpos[:, :dim//2]
            right_qpos = qpos[:, dim//2:]
            
            left_cmd, right_cmd = None, None
            if actions is not None and len(actions) == num_steps:
                 act_dim = actions.shape[1]
                 left_cmd = actions[:, :act_dim//2]
                 right_cmd = actions[:, act_dim//2:]
            
            # 保存路径
            save_name = file_path.stem + "_dynamics.png"
            save_path = file_path.parent / save_name
            
            DynamicsFunction.plot_and_save(
                timestamps,
                left_qpos,
                right_qpos,
                save_path,
                title_suffix=f"\nSource: {file_path.name} @ {fps}Hz, horizon={action_horizon}",
                left_cmd=left_cmd,
                right_cmd=right_cmd,
                action_horizon=action_horizon,
            )
            
    except Exception as e:
        print(f"❌ 分析出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化机械臂速度和加速度")
    parser.add_argument("file", nargs="?", help="要分析的 HDF5 文件路径")
    parser.add_argument("--dir", help="批量分析指定目录下的所有 .hdf5 文件")
    parser.add_argument("--fps", type=int, default=30, help="控制频率 Hz（默认30）")
    parser.add_argument("--action_horizon", type=int, default=15, help="Action chunk 大小（默认15）")
    
    args = parser.parse_args()
    
    if args.file:
        analyze_hdf5(args.file, fps=args.fps, action_horizon=args.action_horizon)
    elif args.dir:
        target_dir = pathlib.Path(args.dir)
        if target_dir.is_dir():
            files = list(target_dir.glob("*.hdf5"))
            print(f"🔎 发现 {len(files)} 个 HDF5 文件")
            for f in files:
                analyze_hdf5(f, fps=args.fps, action_horizon=args.action_horizon)
        else:
            print(f"❌ 目录不存在: {args.dir}")
    else:
        print("ℹ️ 使用说明:")
        print("  1. 离线分析: python plot_dynamics.py <path_to_hdf5_file>")
        print("  2. 批量分析: python plot_dynamics.py --dir <directory_with_hdf5>")
        print("\n  要集成到 main_dual.py，请在代码中添加:")
        print("    from plot_dynamics import RobotStatePlotter")
        print("    runtime = ... subscribers=[RobotStatePlotter(), ...] ...")
