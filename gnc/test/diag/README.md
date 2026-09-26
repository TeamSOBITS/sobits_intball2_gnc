# 障害物回避のsolveの診断

`replan_minco`の障害物込みのsolve（`minco_native_py.plan_minco(grid=...)`）が失敗する理由を追うための道具。

流れ: 段階5のJEMの場面（`../experiment_obstacle_stage5_emergency_stop.py`）を回して`plan_minco`の引数を保存し、
保存した引数だけでsolveを何度でも再現する（同じ引数なら毎回同じ結果）。場面を回すのは保存のときだけ。

## 実行

リポジトリのどこからでも実行できる。`sobits_intball2_gnc`が見えるように`PYTHONPATH`に`gnc/`を入れる:

```bash
source /root/colcon_ws/install/setup.bash
export PYTHONPATH=/root/colcon_ws/src/sobits_intball2_gnc/gnc:$PYTHONPATH
```

1本ずつ順に回すこと。罰則の計算はOpenMPで8スレッド決め打ち（`minco_solver.cpp`の`PENALTY_LOOP_THREADS`）なので、
並列にするとコアを取り合って20倍前後遅くなる（単独なら場面1本が十数秒）。

## スクリプト

| スクリプト | すること |
|---|---|
| `capture_rest.py <ahead_m> <inflation_m> <out.json> [--mid-x=...]` | 非常停止のあと最初の静止からの再計画で、trackerの形のsolveと、初期値の中点のxを決め打ちしたsolveを保存 |
| `capture_flight.py <ahead_m> <appear_after_m> <inflation_m> <out.json>` | 人が出てから最初の非常停止までの、走行中の形のsolveを全部保存 |
| `replay_trace.py <capture.json> [label ...]` | 保存したsolveを1本ずつ解き直し、結果の軌道が膨張した格子に入る所（かすめ）を出す。`MINCO_REBOUND_TRACE=1`でC++の経過も出る |
| `replay_all.py <tag> <capture.json> [...] [--shift-start=DY]` | 保存した全solveを解き直し、error_codeとかすめの数を集計（clean = error_code 0でかすめなし）。`--shift-start`で始点を−yへずらす |
| `constraint_points.py <capture.json> <label>` | 1本の結果の制約点それぞれが膨張した格子に入るかと、かすめの時間を並べる（衝突が制約点の間かを見る） |
| `async_paced_scenario.py <ahead_m> <appear_after_m> [--sync-on-collision]` | 段階5のJEMの人の場面を、本番と同じ非同期の再計画で、sim時間を実時間と同じ速さ（RTF=1）で進めて回す |
| `free_gaps.py <capture.json> [--z=5.0]` | 保存した箱の位置で、JEMを横切る空きの幅を膨張0.1m・0.2mで格子から測る |
| `trace_scenario.py <ahead_m> <appear_after_m> [--sync] [--latency=S] [--proto]` | `async_paced_scenario.py`に、再計画ごとの出発点・膨張の中か・箱との余裕・成否と、箱の横のすき間の幅の表示を足したもの |
| `repro_run4.py [--latency=S] [--appear-clr=M] [--proto [--no-b1] [--no-b2] [--no-b4]]` | シムの走行4（逆向き、古い地図のsolveが走っている最中に箱が出て、止まり始めが遅れる）の再現 |
| `rest_near_box.py <box_x> <box_y> <box_z> <clearance_m> [--reverse]` | 箱の手前に静止した状態から再計画させる（出発点が膨張の中だと解けないことの確認） |
| `early_stop_tracker.py` | 試作（本番に入れていない）: `ReplanMincoTracker`を継承し、待たずに止める（B-1）・届いたsolveを最新の地図で判定（B-2）・止まる最中に解けたら乗り換える（B-4）を足したもの。`docs/2026-09-26_obstacle_emergency_stop_and_recovery.md`の5節 |

`diag_common.py`は共通の処理（保存・読み込み、格子地図の組み立て、かすめの数え方）。
保存したファイル（JSON）はリポジトリに入れず、scratchなど外に置く。

## C++の診断用の切り替え（`minco_native_py/src/`、既定の動きは変えない）

| 環境変数・マクロ | 効果 |
|---|---|
| `MINCO_REBOUND_TRACE`（設定するだけ） | 細かい確認・粗い確認の衝突区間、A*の成否と経路の範囲、対の中身、各段のlbfgsの戻り値、終わった理由をstderrへ |
| `MINCO_DIAG_MAX_RESTARTS=N` | 細かい確認でのやり直しの上限（既定3、EGO v2の`restart_nums < 3`） |
| `MINCO_DIAG_CHECK_STEP_DIVISOR=D` | 細かい確認の刻み`res/D/max_vel`（既定2、1でEGO v2どおり） |
| `MINCO_DIAG_BOUNDS="xmin,ymin,zmin,xmax,ymax,zmax"` | 範囲の外を膨張した格子の中として扱う |
| ビルド時のマクロ`MINCO_CPS_PER_PIECE` | 1区間の制約点の数（`CONSTRAINT_POINTS_PER_PIECE`）。例: `colcon build --packages-select minco_native_py --build-base <dir>/build --install-base <dir>/install --cmake-args -DCMAKE_CXX_FLAGS=-DMINCO_CPS_PER_PIECE=15`で別の場所にビルドし、`PYTHONPATH`の先頭に`<dir>/install/minco_native_py/local/lib/python3.10/dist-packages`を入れる |

## 読み方の例

```bash
MINCO_REBOUND_TRACE=1 python3 replay_trace.py cap.json own_seed 2>&1 | grep "fine: nPoints\|pairs\|END\|RESULT"
```

- `solve: END ... restarts=3 stillUnsafe=1`: 細かい確認で衝突が見つかり続け、やり直しの上限に達した
- `pairs: corner seg(a,a+1) ... dist=0.0000`と`gotIntersectionId=-1`: 隣り合う制約点の間のかすめで、対が足されない（EGO v2の1cmの条件）。以降同じ解を繰り返す
- `lbfgs=-1011`（`LBFGSERR_MINIMUMSTEP`）と`T=[0.01, ...]`: ぶつかったままの区間の時間が潰れた
- `astar: adjustStartEnd FAIL ... endOcc=1`: A*の端点が膨張の中にあり、空きを探す向きが壁に沿って探索範囲の外に出た
