from abc import ABC, abstractmethod

class BasePlanner(ABC):
    """経路計画の抽象インターフェース."""

    @abstractmethod
    def plan(self, start, goal, collision_checker, **kwargs):
        """start→goal の経路を計算し、ウェイポイントリスト [(x,y,z), ...] を返す.

        Args:
            start: 始点座標 (x, y, z)
            goal: 終点座標 (x, y, z)
            collision_checker: 衝突判定オブジェクト
            **kwargs: アルゴリズム固有のオプション（max_iter, timeout, goal_sample_rate等）

        Returns:
            list: ウェイポイントのリスト [(x,y,z), ...]。到達不可能な場合は空リスト。
        """
        pass