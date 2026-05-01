import rospy
from .collision_checker import CollisionChecker
from .base_planner import BasePlanner
from .astar_planner import AStarPlanner
from .safety_astar_planner import SafetyAwareAStarPlanner
from .smoother import shortcut, push_from_walls, interpolate_path
from .visualize import publish_path
from .obstacle_manager import ObstacleManager
from .path_planner import PathPlanner, insert_virtual_waypoints


def check_remaining_path(path, start_index, cc):
    """残りパスの全セグメントを検証し、最初の衝突セグメントのインデックスを返す.

    Args:
        path: ISS座標のウェイポイントリスト（positionのみ、タプル非対応）
        start_index: 検証開始インデックス
        cc: CollisionChecker

    Returns:
        衝突があった場合はそのセグメント開始インデックス、なければ None
    """
    for j in range(start_index, len(path) - 1):
        if not cc.check_line(path[j], path[j + 1]):
            return j
    return None


def apply_path_filters(path, collision_checker, filters=None, **kwargs):
    """
    経路に対して指定されたフィルタ（後処理）を順番に適用するパイプライン関数.

    Args:
        path: ウェイポイントのリスト
        collision_checker: 衝突判定オブジェクト
        filters: 適用する関数のリスト (None の場合はデフォルト設定を適用)
        **kwargs: 各フィルタ関数に渡す共通引数 (push_step, max_iter など)

    Returns:
        加工済みのウェイポイントリスト
    """
    if not path:
        return []

    if filters is None:
        filters = []
        rospy.logwarn("apply_path_filters: no filters specified, returning path as-is")

    processed_path = list(path)
    
    for filter_func in filters:
        try:
            # 各フィルタは (path, collision_checker, **kwargs) の形式で呼び出し可能
            processed_path = filter_func(processed_path, collision_checker, **kwargs)
        except Exception as e:
            # フィルタ名取得の安全策
            func_name = getattr(filter_func, '__name__', str(filter_func))
            rospy.logerr(f"Filter {func_name} failed: {e}")
            
    return processed_path