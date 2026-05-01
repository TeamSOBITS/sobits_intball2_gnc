#!/usr/bin/env python3
"""Path Planner で経路を生成し、ロボットを実際に移動させる."""
import os
import sys
import argparse
import yaml
import rospy
import tf2_ros

# 1. 自作スクリプトがある場所を sys.path に追加
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# 2. LD_LIBRARY_PATH をチェックして再起動
local_lib = os.path.expanduser("~/.local/lib")
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
if local_lib not in ld_path:
    os.environ["LD_LIBRARY_PATH"] = local_lib + (":" + ld_path if ld_path else "")
    try:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception as e:
        print(f"Failed to re-execute: {e}")
        sys.exit(1)

from navigation.tf_frame_resolver import wait_for_tf
from navigation.pose_resolver import PoseResolver
from control import ActionExecutor, SmoothActionExecutor
from control.trajectory_follower import TrajectoryFollower
from navigator import Navigator
from guidance import ObstacleManager
from guidance.path_planner import PathPlanner
from gnc_defaults import GNC_DEFAULTS


class GNCManager:
    def __init__(self):
        # gnc_params.yaml のロード（必ず最初に実行する）
        self._load_gnc_params()

        # 共有リソースの初期化
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)

        # TF が利用可能になるまで待機
        if not wait_for_tf(self.tf_buffer, timeout=10.0):
            rospy.logwarn("TF not available at init, proceeding anyway")

        # --- Executor の切り替え（YAML の executor_mode から決定）---
        executor_mode = rospy.get_param('/gnc/executor_mode', GNC_DEFAULTS['executor_mode'])
        if executor_mode == 'smooth':
            self.executor = SmoothActionExecutor()
            rospy.loginfo("GNCManager: Using SmoothActionExecutor (smooth mode)")
        elif executor_mode == 'steady':
            self.executor = ActionExecutor()
            rospy.loginfo("GNCManager: Using ActionExecutor (steady mode)")
        else:
            rospy.logerr("GNCManager: Unknown executor_mode '%s'. Valid: steady / smooth", executor_mode)
            rospy.signal_shutdown("Invalid executor_mode")
            return

        # ObstacleManager の初期化
        obstacle_topic = rospy.get_param('/gnc/obstacle_topic', GNC_DEFAULTS['obstacle_topic'])
        self.obstacle_manager = ObstacleManager(self.tf_buffer, topic=obstacle_topic)
        rospy.loginfo("GNCManager: ObstacleManager initialized (topic=%s)", obstacle_topic)

        # 専門クラスの初期化（DI）
        self.path_planner = PathPlanner()
        self.pose_resolver = PoseResolver(self.tf_buffer, self.executor)
        self.trajectory_follower = TrajectoryFollower(
            self.tf_buffer, self.executor, self.path_planner,
            obstacle_manager=self.obstacle_manager)

        # Navigator の初期化（指揮者：各専門クラスに委譲）
        self.navigator = Navigator(self.tf_buffer, self.executor,
                                   obstacle_manager=self.obstacle_manager,
                                   path_planner=self.path_planner,
                                   pose_resolver=self.pose_resolver,
                                   trajectory_follower=self.trajectory_follower)

    @staticmethod
    def _load_gnc_params():
        """gnc_params.yaml を ROS パラメータサーバーにロードする."""
        import rospkg
        try:
            pkg_path = rospkg.RosPack().get_path("sobits_intball2_gnc")
        except rospkg.ResourceNotFound:
            pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        yaml_path = os.path.join(pkg_path, "config", "gnc_params.yaml")
        if os.path.isfile(yaml_path):
            with open(yaml_path) as f:
                params = yaml.safe_load(f)
            if params:
                for ns, values in params.items():
                    if isinstance(values, dict):
                        for k, v in values.items():
                            rospy.set_param("/{}/{}".format(ns, k), v)
                    else:
                        rospy.set_param("/{}".format(ns), values)
            rospy.loginfo("GNCManager: Loaded gnc_params.yaml from %s", yaml_path)
        else:
            rospy.logwarn("GNCManager: gnc_params.yaml not found at %s, using defaults", yaml_path)

    def navigate(self, **kwargs):
        """Navigator に委譲する."""
        return self.navigator.navigate(**kwargs)

# --------------- Main ---------------

def parse_args():
    parser = argparse.ArgumentParser(description="Path Planner で経路を生成しロボットを移動させる")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", type=str, help="目的地の TF フレーム名")
    group.add_argument("--goal", nargs=3, type=float, metavar=("X", "Y", "Z"), help="目的地の iss_body 座標")

    parser.add_argument("--offset", nargs=3, type=float, default=[0, 0, 0], help="オフセット")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rospy.init_node("gnc_manager", anonymous=True)

    # クラスのインスタンス化（動作パラメータはすべて gnc_params.yaml から読み込む）
    gnc = GNCManager()

    # メソッド呼び出し
    success = gnc.navigate(
        target_frame=args.target,
        goal=args.goal,
        offset=args.offset,
    )

    if success:
        rospy.loginfo("Final result: SUCCESS")
    else:
        rospy.logerr("Final result: FAILED")

if __name__ == "__main__":
    main()