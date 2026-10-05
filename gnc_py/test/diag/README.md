# 障害物回避のsolveの診断

`replan_minco` の障害物込みのsolve（`sobits_intball2_gnc_cpp.plan_minco(grid=...)`）が失敗する理由を追うための道具です。

流れ: 段階5のJEMの場面（`../experiment_obstacle_stage5_emergency_stop.py`）を回して `plan_minco` の引数を保存し、
保存した引数だけでsolveを何度でも再現します（同じ引数なら毎回同じ結果）。場面を回すのは保存のときだけです。

- `capture_*.py`: 場面を回して引数を保存する
- `replay_*.py`、`constraint_points.py`、`free_gaps.py`: 保存した引数から解き直して調べる
- `*_scenario.py`、`repro_run4.py`、`rest_near_box.py`: 場面そのものの再現
- `diag_common.py`: 共通処理

使い方は各スクリプトの先頭のdocstringを見てください。保存したJSONはリポジトリに入れず、外に置きます。

## 守ること

- 実行前に `source /root/colcon_ws/install/setup.bash` し、`PYTHONPATH` に `gnc_py/` を入れる。
- 1本ずつ順に回す。罰則の計算はOpenMPで8スレッド決め打ちなので、並列にするとコアを取り合って大きく遅くなる。
- C++側の診断用の切り替えは、環境変数 `MINCO_REBOUND_TRACE` と `MINCO_DIAG_*`（`gnc_cpp/src/guidance/minco/`）。既定の動きは変えない。
