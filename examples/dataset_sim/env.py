from typing import Optional, Dict, Any

import numpy as np
from typing_extensions import override

try:
    # OpenPI environment base (for API compatibility)
    from openpi_client.runtime import environment as _environment
except Exception:
    # Fallback: minimal base class if openpi_client not available
    class _BaseEnv:
        def reset(self) -> None: ...
        def is_episode_complete(self) -> bool: ...
        def get_observation(self) -> Dict[str, Any]: ...
        def apply_action(self, action: Dict[str, Any]) -> None: ...
        def close(self) -> None: ...
    _environment = type("environment", (), {"Environment": _BaseEnv})

# LeRobot dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class DatasetSimEnvironment(_environment.Environment):
    """基于LeRobot数据集的简洁仿真环境。

    - reset(): 重置到指定episode的第一帧
    - get_observation(): 返回数据集中的observation字段
    - apply_action(action): 与数据集中的action对比并记录差异
    - is_episode_complete(): 当前episode是否结束
    """

    def __init__(
        self,
        dataset_path: str,
        episode_index: int = 0,
        step_stride: int = 1,
    ) -> None:
        """初始化环境。

        Args:
            dataset_path: LeRobot数据集路径。
            episode_index: 启动时选择的episode索引。
            step_stride: 每次动作后的帧步进（默认1）。
        """
        self.dataset = LeRobotDataset(dataset_path)
        self.episode_index = max(0, min(episode_index, self.dataset.num_episodes - 1))
        self.step_stride = max(1, int(step_stride))

        self._done = True
        self._cur_idx = 0
        self._from_idx = 0
        self._to_idx = 0

        # 记录最近一次动作差异
        self.last_action_diff: Optional[Dict[str, Any]] = None

        # 直接准备episode边界，reset时激活
        self._set_episode_bounds(self.episode_index)

    def _set_episode_bounds(self, ep: int) -> None:
        epi = self.dataset.episode_data_index
        self._from_idx = int(epi["from"][ep].item())
        self._to_idx = int(epi["to"][ep].item())

    @override
    def reset(self) -> None:
        """重置到当前episode的第一帧。"""
        self._set_episode_bounds(self.episode_index)
        self._cur_idx = self._from_idx
        self._done = False
        self.last_action_diff = None

    @override
    def is_episode_complete(self) -> bool:
        return self._done

    def _current_item(self) -> Dict[str, Any]:
        return self.dataset[self._cur_idx]

    def _get_by_path(self, obj: Any, path: str) -> Any:
        cur = obj
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return cur

    def _to_numpy(self, obj: Any) -> Any:
        # 轻量转换：支持 tensor、numpy、dict、list/tuple
        if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
            return obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, dict):
            return {k: self._to_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_numpy(v) for v in obj)
        return obj

    @override
    def get_observation(self) -> Dict[str, Any]:
        """返回按 LeRobotFlexivDataConfig repack_transforms 组织的 observation。

        结构：
        {
          "images": {
             "cam_left_wrist": observation.images.left_wrist,
             "cam_right_wrist": observation.images.right_wrist,
          },
                    "state": observation.state,
                    "prev_state": observation.prev_state (如果存在)
        }
        所有张量将被转换为 numpy。
        """
        item = self._current_item()

        # 提取图像
        cam_left = item['observation.images.cam_left_wrist']#self._get_by_path(item, "observation.images.cam_left_wrist")
        cam_right = item['observation.images.cam_right_wrist'] #self._get_by_path(item, "observation.images.cam_right_wrist")

        # 提取状态
        state = item['observation.state'] #self._get_by_path(item, "observation.state")
        if state is None and isinstance(item, dict):
            state = item.get("state")

        # 提取 prev_state（用于 pose 模式或需要前一状态的情况）
        prev_state = item['observation.prev_state'] #self._get_by_path(item, "observation.prev_state")
        if prev_state is None and isinstance(item, dict):
                prev_state = item.get("prev_state")
        
        prompt = item['task']
        
        obs: Dict[str, Any] = {"images": {}}
        if cam_left is not None:
            obs["images"]["cam_left_wrist"] = cam_left
        if cam_right is not None:
            obs["images"]["cam_right_wrist"] = cam_right
        if state is not None:
            obs["state"] = state
        if prev_state is not None:
            obs["prev_state"] = prev_state
        if prompt is not None:
            obs["prompt"] = prompt

        return self._to_numpy(obs)

    def _get_expert_action(self, item: Dict[str, Any]) -> Optional[np.ndarray]:
        # 兼容不同字段命名
        if "actions" in item:
            return np.asarray(item["actions"])  # 已是向量或数组
        if "action" in item:
            return np.asarray(item["action"])  # 单步动作
        return None

    @override
    def apply_action(self, action: Dict[str, Any]) -> None:
        """对比传入action与数据集中的action并记录差异。

        期望输入格式：{"actions": np.ndarray | list}
        """
        if self._done:
            return

        item = self._current_item()
        expert = self._get_expert_action(item)

        user_act = action.get("actions")
        if user_act is None:
            raise ValueError("缺少'actions'键，期望格式: {'actions': np.ndarray | list}")
        user_act = np.asarray(user_act, dtype=np.float32)

        if expert is None:
            # 无专家动作时，只记录用户动作信息
            self.last_action_diff = {
                "has_expert": False,
                "user_action_shape": user_act.shape,
            }
        else:
            expert = np.asarray(expert, dtype=np.float32)
            # 对齐形状（取最小维度）
            dim = min(user_act.shape[-1], expert.shape[-1])
            ua = user_act[..., :dim]
            ea = expert[..., :dim]
            diff = ua - ea
            l2 = float(np.linalg.norm(diff))
            mae = float(np.mean(np.abs(diff)))

            self.last_action_diff = {
                "has_expert": True,
                "dim": dim,
                "l2": l2,
                "mae": mae,
                "diff": diff.astype(np.float32),
            }

        # 前进到下一帧
        self._cur_idx += self.step_stride
        if self._cur_idx >= self._to_idx:
            self._cur_idx = self._to_idx - 1
            self._done = True

    def close(self) -> None:
        # 数据集不需要特殊清理，这里留空
        pass
