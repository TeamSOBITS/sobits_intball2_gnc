# control/base_executor.py
import abc

class BaseExecutor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def move_to(self, pos, quat, timeout=60.0):
        """ブロッキング移動（現在のNavigatorが使用）"""
        pass

    @abc.abstractmethod
    def send_goal_async(self, pos, quat):
        """非同期移動（将来の動的回避監視に使用）"""
        pass

    @abc.abstractmethod
    def cancel_all_goals(self):
        """即時停止命令（障害物発見時に使用）"""
        pass
    @abc.abstractmethod
    def update_target(self, pos, quat):
        """現在の目標地点を更新する。特異点対策等のロジックは内部で行う"""
        pass
    @abc.abstractmethod
    def is_reached(self, target_pos, target_quat):
        """到達判定のみを行う（監視ループ用）"""
        pass