# 未達成タスク

**対象**: `sobits_intball2_gnc`（ISS内自由飛行キューブ型ロボット IntBall2 の GNC 実装、ROS2/Humble）

**機体特性**: 無重力環境、8ファンによる全方向駆動（fully-actuated）→ 並進と姿勢を独立に制御できる

## このファイルの書き方（方針）

- **未達成タスクのみ**を書く。達成済みの内容は書かない。達成済みの経緯・調査結果は`docs/archive/achieved/`配下を参照する。
- **Phase番号は使わない**。フラットな優先度順リストとする（上にあるものほど優先度が高い）。
- 各タスクは `[タグ] タスク名` の見出し＋短い動機（1文、目安40字程度）＋必要なら参照リンク、の形式で書く。
  - タグはGNCの役割（G/N/C）または領域（運用/Teleop/将来）を示す目印。分類のためではなく一目で系統がわかるようにするための軽い印。
  - 動機は「なぜ今これが必要か」のみ。経緯・調査の数値的根拠・比較実験の詳細はarchiveに委ねる。
  - **例外**: 未達成の判断そのものに直結する数値・閾値（例: トルク予算の値、角度の閾値）は動機に残してよい。それが無いとタスクの意味が読めなくなるため。
- 「タスク」と「未決定事項（判断待ちの論点でタスクではないもの）」は別セクションに分ける。混ぜない。

---

## 未達成タスク（カテゴリ別・各カテゴリ内は優先度順）

### [G] 微小移動指令のデッドゾーン
5cm未満のような微小指令でもA*→Hermite→TOPP-RAのフルパイプラインが走り、`static_minco`時は3〜5秒ブロックする。閾値未満は即`STATUS_SUCCEEDED`で返す。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（B-2）

### [G] K分離(2自由度版)、短距離レグでのduration悪化が未改良
MINCOの姿勢waypoint密度↑時のduration悪化は圧縮できた（+133%→+21%等）が、短距離レグでは悪化が
残ったまま（`docs/archive/2026-09-01_replan_speedup_options_overview.md`）。要改良。

### [G] attitude_reference_mode=look_at 本体実装
対象TFフレーム名を`guidance.look_at_target_frame`（`ros2 param`）で受ける設計までは確定済みだが、`_run_trajectory`ループ内で毎tick TFルックアップして姿勢を再計算する本体が未着手。現在選択すると`face_travel`にフォールバックし警告ログのみ出す。TFロスト時のフォールバック方針（直前の`q_des`保持 or goal中断）も未決定。

### [G] pre_align と look_at 併用時の事前整列目標方向
現在の実装は`face_travel`時のみ`v_early`方向に事前整列。`look_at`本体実装時に、事前整列先を「初期タンジェント」から「look-at対象方向」に切り替える設計が必要。

### [G] 移動前のロール事前回転（到着後の`align_at_arrival`高速化狙い）
`attitude_reference_mode`(`face_travel`/`look_at`)はピッチ・ヨー（進行方向を向く方向）を経路に応じて決めるが、その向きを軸にした回転（ロール）は決めない。移動中ロールが放置されると到着時に大きなロール誤差が残り、`align_at_arrival`の補正が遅くなる（角度が大きいほど遅く・精度も悪化する、`docs/archive/achieved/2026-08-21_tf_correction_align_slow_investigation.md`のゲイン実測で確認済み）。移動中にロールも回転させると貴重な推力が減ってしまう。そのため、移動前に、到着後のロールだけでも合わせておくことで、事後回転の高速化が狙えるはず。優先度中

### [G] 経路の補間方式（直線移動モード）
waypoint間を滑らかに補間するか、ただの直線でつなぐか未検討。姿勢モードとは直交する軌道生成側の話。`BaseTrajectoryGenerator`に3つ目の実装を追加するか、既存`HermiteSplineTrajectoryGenerator`のパラメータで代替できないか検討する。

### [G] wrench_envelope_safety_marginのさらなる調整余地
`guidance.wrench_envelope_safety_margin=0.7`＋`thrust_allocator.minimax_objective=true`で
duty≥0.95飽和頻度を48%→33.4%まで改善したが、まだ33%残っている（`docs/archive/achieved/
2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その7）。安全係数をさらに下げる
（0.6/0.5）ことで飽和頻度と所要時間のトレードオフを追加探索する余地がある。

### [G] 慣性テンソルの非等方性検証
等方前提だとジャイロ項ω×(Iω)がゼロになり無視できるが、実機の慣性テンソルが非等方ならこの項が復活し角速度の2乗で効いてくる。現状の等方スカラー0.0136 kg·m²が実機と一致するか未検証。

### [G] 抗力トルク実測値（κ_j/k_j）をAに反映して包絡を再構築（シム未対応のため実機フィデリティはシム上で検証不可）
三谷・西下・平野2023の実測値から8基分のκ_j/k_jは同定済み・番号対応も検証済み（ロールで36%の能力回復、力とヨーは正しく1〜2割補正、面数24→112）。実装対象は自分たちの`A`行列（`actuation_envelope.py`）のみでシム変更は不要だが、現在のシムは`kappa=0`（抗力トルクなし）のままなので、シム上ではこの改善による実機フィデリティ向上分を検証しようがない点に注意。付随課題として実機の推力方向ベクトル未入手（公称モデルとの乖離あり、Y方向が実測で3割弱い；著者への問い合わせ候補）。

### [G] JAXAの直線移動を移植したモード（`jaxa_linear`、優先度低）
移動中の制御だけをJAXAと同条件で比べる場合に必要（ホバーだけなら不要）。`gnc_py/sobits_intball2_gnc/common/utils/stopping_profile.py`は`target()`の減速側だけの移植で、加速・巡航側と移動用プロファイル（`v_max`・`w_max`、回転と並進の順序）は未移植。本番モードにせず`gnc_py/test/manual/`のスクリプトにする案もある。詳細: `docs/archive/achieved/2026-09-24_guidance_directory_restructure_plan.md`（手順5）

### [G] replan_minco の非常停止後・大ズレ時に実測から再計画する
再計画・衝突チェック・停止プロファイルがすべて参照状態ベースのため、衝突や強い外乱で機体が参照から大きく外れると実機の危険を見落とす。EGO v2は非常停止復帰時のみ実測から計画し直す（`planFromGlobalTraj`の`odom_pos_`/`odom_vel_`）。まず非常停止後の静止再計画を実測位置から始め、必要なら参照−実測ズレ閾値での実測再計画/停止を足す（`[C] 追従誤差ガード（`Dtc`相当）`と関連）。詳細: `docs/arch/2026-09-28_zeno_phenomenon_in_replan.md`

### [G] 障害物回避: 1.0m先の非常停止と復帰をシムで確認
B-1・B-2・B-4（非同期の衝突時の早い停止と減速中の再計画）の本来の見せ場がシム未確認（2.0m先は2026-09-28に成功）。場面は`nav_entry`→`inspection_entry_1`、0.8m進んだら1.0m先に箱。詳細: `docs/archive/achieved/2026-09-26_obstacle_emergency_stop_and_recovery.md`

### [G] 障害物回避: 膨張の中から抜け出す処理と、止まる位置が膨張に入るのを防ぐ
入り込むと静止から解けず止まったままになる（A）。`StoppingProfile`は障害物を見ないので、強い減速（B-3）かプロファイル側での回避が要る。1.0m先のシム結果を見てから決める。詳細: `docs/archive/achieved/2026-09-26_obstacle_emergency_stop_and_recovery.md`

### [G] 障害物回避: 押し出す先の壁を考えない・成功扱いの軌道にかすめが残る
EGO v2の対はぶつかった障害物の分しか付かず、JEMの場面では成功扱いの軌道の17本中7本に膨張格子のかすめが残り、採用後の非常停止の原因になる。詳細: `docs/archive/achieved/2026-09-24_obstacle_avoidance_jem_map_check.md`

### [G] guidance_node終了時に非同期solveが走っていると落ちるか確認
単体テストでは、C++の中にいるdaemonスレッドのままPythonが終わると`Fatal Python error: Aborted`で落ちた。本番で同じか未確認。詳細: `docs/archive/achieved/2026-09-26_obstacle_emergency_stop_and_recovery.md`（7節）

### [G] MINCOの罰則計算のOpenMPスレッド数が8に決め打ち
`gnc_cpp/src/guidance/minco/objective.cpp`の`PENALTY_LOOP_THREADS`。CPUを取り合うとsolveが20倍前後遅くなり、1s周期を超えうる（シムの走行4で最大0.87s）。詳細: `docs/archive/achieved/2026-09-24_obstacle_avoidance_jem_map_check.md`

### [G] replan_minco: 経由点を飛ばす問題の新設計と、globalの時間配分の修正
A*が経由点を出し始めると往復・周回ルートが壊れる。経由点ごとにglobalを作り直す設計は決定済み・実装は保留、`heuristicSegmentTimes`（角での減速の見積もり）の修正と一緒に行う。詳細: `docs/archive/achieved/2026-09-24_replanning_minco_v3_waypoint_skip_offline_check.md`

### [G] replan_minco: globalとlocalの速度上限を1つにそろえる
global（`target_speed` 0.5）とlocal（`minco_local_max_vel` 0.15、face travel時のみ）が別々で、globalが上限超えで経由点を通るとlocalが解けない。詳細: `docs/archive/achieved/2026-09-24_replanning_minco_v3_waypoint_skip_offline_check.md`

### [G] replan_minco: 失敗時の即時再試行と、イベント駆動の再計画（EGO v2 A3・A4）
再計画のきっかけが固定周期（1s）だけで、失敗すると次の周期まで待つので障害物への反応が遅れる。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（A3・A4）

### [G] face travelなしでも機体座標のwrench評価を入れるか
face travelなしでは`body_frame_wrench`を渡しておらず、単位姿勢でない出発では世界座標の評価が最大1.42倍ずれる。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§6）

### [G] replan_minco: 出発直後と姿勢合わせ中のduty飽和
出発後0〜10sに飽和が集中し、姿勢合わせ中は30〜60%。JEM経路の走行1では最大遅れ241mm・飽和44.1%で、原因を回避オフで同じ経路を走って切り分ける。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§2・§11）、`docs/archive/achieved/2026-09-24_obstacle_avoidance_production_integration_plan.md`

### [G] replan_minco: 経由点の角のショートカット量と、カーブさせるglobalの検証
狭い場所では丸めた角が壁に近づくおそれがある。障害物コストを入れた後の再評価と、経由点あり＋カーブの設定でのシム検証。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§4・§12）

### [G] replan_minco: 別のルート（大きく曲がる・3D）でのシム検証
face travel・障害物回避のシムは`nav_entry`↔`inspection_entry_1`の1ルートだけで、ほかの形は動作未確認。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§5）

### [G] min snap のコアロジック実装（`min_snap_trajectory_generator.py`、別担当者、優先度低）
Phase 2で契約は確定済みだがコア実装が未着手。pure functionとして実装（ROS import禁止）。理論: Mellinger & Kumar (2011)。入出力契約: `docs/minimum_snap/min_snap_interface_contract.md`。参考実装: https://github.com/The-SS/quadrotor_trajectory 、参考解説: https://dev10110.github.io/tech-notes/control-theory/min_snap.html
成果物: `test_min_snap.py`に数値解検証テスト追加、`test_trajectory_generator_contract.py`の対象に`MinSnapTrajectoryGenerator`追加、区間時間→軌道のパイプラインの統合テストを追加して確認。依存関係なし（他タスクと並行可）。`static_toppra`モードは既にTOPP-RA（`ToppraTrajectory`）で力/トルク制約を考慮した軌道生成に置き換わっているため、緊急度は下がっている。

### [C] 姿勢FFの既定ON化判断（残りの検証）
`trajectory_controller.attitude_feedforward`は既定`false`のまま。`static_toppra`の1ルートではON時に移動中の姿勢誤差が最大3.23°→0.51°だったが、新コードでのOFF再計測と`static_minco`・`replan_minco`での検証が未実施。これらを済ませてから既定をONにするか決める。詳細: `docs/archive/achieved/2026-09-23_attitude_feedforward_implementation_and_sim_verification.md`

### [C] omega_errを数値微分から解析値へ置き換え
現状50Hzの数値微分（`att_filter_alpha=1.0`で無フィルタ）でノイズと位相遅れを抱えている。姿勢FF導入（2026-09-23、`docs/archive/achieved/2026-09-23_attitude_feedforward_implementation_and_sim_verification.md`）で`w_des`が取れるようになったため、`omega_imu - R(qe)*w_des`で解析的に算出できる。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（A-2）

### [C] thrust_allocatorの決定的な配分フォールバック
`lsq_linear`（反復ソルバ、計算時間が非決定的）に時間上限を設け、超過・失敗時はJAXA型の行列配分（事前計算行列の積と最小値減算のみ、固定時間）に落とす。詳細: 同上（B-4）

### [C] 追従誤差ガード（`Dtc`相当）の追加
追従誤差が閾値を超えても軌道を中断しない。特に`static_toppra`・`static_minco`は開ループのため機体位置に関わらず基準時刻が進み続け、外乱・衝突を検知できない。閾値超過で中断時制動（`GuidanceExecutor.brake()`、`docs/archive/achieved/2026-09-24_cancel_stopping_profile_implementation_and_sim_verification.md`）経由のholdへ落とすガードが必要。詳細: 同上（C-1）

### [C] trajectory_controller のTF速度推定ノイズ調査
move_to中に`f_des`が瞬間的に0.68N超まで跳ねる事象を観測、計画側（`a_des`/`v_des`）はほぼ無風
だったためフィードバック起因（`kd_pos`側）と特定済みだが、`trajectory_controller`のTF速度推定
（差分→`vel_filter_alpha`のEMA）自体のノイズ・遅延特性の単体調査は未着手
（`docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その8）。
同じスタンプが続くときの速度0バイアスは修正済み（77c7f98）。数ms以内の近いTFペアで速度が跳ねる件（`trajectory_controller.py`の`stamp - self._last_t > 1e-6`だけの判定）と、`pose_corrector.py`の同時刻TFで速度0をEMAへ入れる件が残る。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§1・§13）

### [C] trajectory_controller.max_torque/thrust_allocator.torque_weight_ref の理論値への再設計
`kp_att`/`kd_att`は`tf_correction`の検証済み値（0.20）に合わせて再チューニング済み
（`docs/archive/achieved/2026-08-27_trajectory_controller_torque_redesign_plan.md`、
`docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その4）。
残っているのは`max_torque`（現状0.32Nm）・`thrust_allocator.torque_weight_ref`（同0.32Nm）を
理論トルク上限（x:0.00303, y:0.00455, z:0.00819 Nm）に近づける再設計。ただし実際の出力トルクは
`thrust_allocator`のLSQ解が決めており、このソフトクランプ自体がボトルネックである可能性は低く
優先度は低め。`torque_weight_ref`を理論値に近づける場合は`force_weight_ref`とセットで、
2026-08-20のforce-crush修正（`docs/archive/achieved/2026-08-20_thrust_allocator_force_crush_fix.md`）
の重み付けバランスを崩し再発させないか同じ形式の再検証が必要。

### [C] tf_correction.kd_pos の位置ホールド時ノイズ増幅の見直し（優先度低）
静止ホールド中にpx 1.0mm/pz 4.0mm peak-to-peak、周期8-9秒の微小な振動を確認。同じ仕組みの`kd_att_hold`（TF有限差分角速度ノイズの増幅）を半減して振幅が約1/11に減った実績があり、`kd_pos`（TF有限差分速度）も同様の見直しで改善する可能性がある。ただし周期が姿勢側(0.67秒)よりだいぶ遅く、`smooth_window`のTF平滑化による位相遅れなど別要因が絡む可能性もあり未検証。

### [C] 積分項の追加（実験・優先度低）
JAXAは`ki`と飽和付き積分器（`fi_max=0.02`）が配線済み（現状`ki=0`）だが、SOBITSには無い。重心オフセット・配分行列誤差由来の定常バイアス切り分け材料として、`ki=0`から始めればリスクなく実験できる。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（B-3）

### [N] 実機用の自己位置・姿勢推定（IMU単独）
実機にTFは存在しない（TFはシム限定のオラクル）。並進はIMU二重積分で誤差が時間の2乗で発散し長時間精度維持が原理的に困難。姿勢はジャイロ一重積分でより現実的だが、無重力下では加速度計による重力基準補正が使えない。`navigation/utils/`を新設する方針で合意済みだが着手時期・優先度は未定。詳細: `docs/archive/achieved/2026-08-19_phase0_findings.md`（観測13）

### [障害物] 出発時に最初のローカルが交差 → 停止状態からの解き直しで暴れる
`initial_local_collides`で出発を止めた後、停止状態からの解き直し（ランダムな初期形状）で長さ14 m・z=1.43 mまで潜るローカルが1回採用された（ほぼ動く前に置き換わり実害は小）。関連: `test_obstacle_map_depth.py::test_first_local_through_a_depth_seen_obstacle_starts_held`が失敗（修正で最初のローカルが突き抜けなくなり前提の状況が作れない、テストの作り直しが必要）。経緯は`archive/achieved/2026-09-28_gpl_derived_code_rewrite_plan.md`の末尾

### [運用] move_toのgoal responseがタイムアウトする
ゴール受付中に最初のローカル計画（約1 s）を解いている間に`failed to send response (timeout)`、クライアントが結果を受け取れない（09-29に3回）

### [運用] シム/bridge/gnc_bringup起動順序によるホバー保持不能の再発調査
シム・ROS1↔ROS2ブリッジ・`gnc_bringup.launch.py`の起動順序やタイミングのズレが原因と思われる、ホバー保持ができなくなる現象が複数回再発している（`/ctl/duty`のforeign publisher競合は原因ではないと確認済み）。`control_node`再起動で復帰することは確認済みだが（ただし`control_node`だけの再起動でも機体が流れて回転した例がある、`docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`§10）、根本原因（起動順・タイミング依存の何か）は未特定。再現条件の特定と恒久対策が必要。TFのデータや時間が汚染され自己位置が汚染される可能性。シム起動→bridge起動→ホバー制御（`control.launch.py`）起動、のタイミングが早すぎる（TF/センサーデータが安定する前に`control_node`が動き出す）と、機体が急速旋回し続ける現象を確認。再現条件の有力候補: `control_node`（ホバー）起動済みの状態でシム・bridgeを再起動すると発生する。各起動ステップ間に十分な待機・データ安定確認を挟む運用ルールが必要。

### [運用] 障害物回避時の壁との余裕を格子地図で測る
箱を避けて-x側などへ回り込むと壁に近づく可能性があるが、壁との余裕を測るスクリプトも記録もない。詳細: `docs/archive/achieved/2026-09-24_obstacle_avoidance_production_integration_plan.md`

### [運用] 単体テストの点検で残った項目
kdの符号（速度誤差≠0）、本番のminimax重み、HoverControllerの上限、setpointの`_callback`、打ち切り300秒、ROSなし環境でのtest collection失敗など（10・16は対応済み）。詳細: `docs/archive/achieved/2026-09-26_unit_test_audit.md`（未完了）

### [運用] replan_minco関連の後片付け
削除済みの`ReplanningTrajectoryTracker`を前提にした説明が`minco_trajectory.py`・`toppra_trajectory.py`・`model_kf_estimator.py`のdocstringに残る。`gnc_py/test/experiment_v3_face_travel_*.py`の試作7本、`gnc_py/test/diag/early_stop_tracker.py`と各スクリプトの`--proto`（本番に入ったので不要）の削除判断。詳細: `docs/archive/achieved/2026-09-23_replanning_minco_v3_remaining_tasks.md`（§15）
### [運用] ROS1↔ROS2ブリッジの本番トピック構成方針
GNC最小構成（`/clock`・`/tf`・`/tf_static`・`/imu/imu`・`/gnc/body_pose_raw`等）に絞るだけで`/tf`の負荷耐性が大幅改善することは確証済み。これを正式運用にするか、`bridge_topics.yaml`を用途別複数用意する仕組みにするか、方針を固める必要がある。詳細: `docs/archive/achieved/2026-08-19_recording_cpu_load_control_degradation.md`

### [運用] `/root/bridge/`の未使用ファイルの後片付け
`bridge_topics_tf.yaml`、`bridge_topics.yaml.bak_*`（4つ）が残っている。要否を判断して消す（コード内の一時デバッグ計装は撤去済み）。
### [運用] 緊急停止/中断サービス（/gnc/stop）の要否（優先度低）
`/gnc/advance_checkpoint`と同じ`std_srvs/Trigger`パターン。Action自体は`cancel_goal`で個別の移動を中断でき、cancel時の制動も実装済み（9c06e8e）のため、それと独立した「今すぐ全部止める」手段が本当に必要か未検討。

### [Teleop] 手動操縦の最大速度の実測
手動操縦コマンドで機体が到達しうる最大並進/回転速度を実測する。

### [Teleop] 緊急ブレーキ機能の要否・実装方針
手動操縦中に急停止させる機能。`/gnc/stop`検討と合わせて整理する。

### [Teleop] 停止性能の実測
最大速度からの停止までの最長時間・最長距離を実測し、緊急ブレーキ機能や安全マージン設計の根拠にする。

### [将来] 仮想カメラに実物のステレオの効果を入れる
理想カメラ（`virtual_camera.launch.py`）は完成・シム検証済み。実物に近づけるため、実測で分かった効果を1つずつ入り切りできる形で足す: 近すぎると穴でなく奥に出る（0.18mで58%が奥）、端の欠け（左32px・右約100px）、z²に比例する距離の誤差、穴、暗い床の偽の手前の点、縁の飛び点、周期・遅延（実物は約3Hz）。あわせて障害物の何割が写ったかのログ。詳細: `docs/archive/achieved/2026-09-26_virtual_obstacle_sensor_requirements.md`「実物の効果」、根拠は同じ場所の`2026-09-28_stereo_camera_measurement.md`

### [将来] 計画側で仮想カメラの深度を使う
guidance_nodeは今も`/guidance/virtual_obstacles`（箱をそのまま）を読んでおり、視野の外・陰・裏側まで分かっている。`/virtual_camera/<camera>/depth`を受けて格子に投影する形に変える: `ObstacleMap`を点で受ける、記憶（ずっと覚える／見通せたら消す／N秒で忘れる、既知の地図と見えた点の層を分ける）、`OccupancyGrid`に消す機能（作り直すかC++に足すか）、地図が10〜30Hzで変わる前提でのtrackerの解き直しの扱い、未観測を空きとみなすか。検出距離（既定0.25〜3.0m）と止まれる距離（約0.65m）の関係も確かめる。方針はEGO v2と同じ（log-odds＋レイキャスト）で、既知の地図の層・ロック・固定範囲の3点だけ変える。詳細: `docs/archive/2026-09-28_virtual_camera_depth_mapping_plan.md`

### [将来] MPCC姿勢/トルク統合の実行可能性課題
並進のみのMPCCはprogress stall解決済み・強擾乱250〜1000tickでinfeasible/予算超過ゼロを確認済み。しかし姿勢/トルク統合プロトタイプでは、弱擾乱時にACADOS_MINSTEPで解が不可解になる、強擾乱時は並進と姿勢が8ファンの推力予算を奪い合い並進収束が4mm→407mmへ悪化する、という新課題が判明。ソルバ時間も13〜15ms/tickに増加（100ms予算内ではある）。本番導入するか自体が未定。詳細: `docs/archive/2026-08-29_mpcc_attitude_torque_integration_plan.md`

### [将来] 並進・姿勢分離の妥当性を数値で正当化
Watterson, Smith & Kumar (IROS 2016)は「全軸の推力能力が同程度なら分離が妥当」と明言しているが、Int-Ball2の包絡は方向によって約58%の開き（`|b|`が0.002516/0.002835/0.003969の3種類）があり、この前提が成立していない可能性がある。定量化できれば位置と姿勢を統合的に最適化する設計判断の根拠になる（論文化する場合は新規性の主張の中核にもなり得る）。

### [将来] 急カーブでの定量比較データ取得（論文用、動作可否は確認済み）
TOPP-RA staticでどんな急カーブでも曲がれること自体は確認済みで、これは動作可否の課題ではない。残るのはIAC-22（振幅0.4m・長さ6.0m・ウェイポイント間隔0.5m）と同条件から曲率をきつくしていき、先読み距離方式が破綻する点との対比データを取る作業で、論文執筆時にのみ必要。ISS船内でRRT*が出す回避経路が実際にどの程度の曲率を要求するかは別途確認が要る。

---

## 未決定事項（判断待ちの論点、タスクではない）

- [運用] 姿勢のわずかな追従ズレへの対応方針: 力優先修正の副作用でトルク予算が約4割減り、姿勢追従に軽微なズレが残る。配分アルゴリズム側の改善余地はほぼ無いと確認済み。「現状維持（ファン物理上限由来として許容）」か「力の精度を意図的に数%犠牲にするトレードオフを試す」かの判断が必要。詳細: `docs/archive/achieved/2026-08-20_thrust_allocator_force_crush_fix.md`
- [運用] デッドレコニング（安全弁）の実装要否: TF停滞時に最後の推定速度で位置を前方外挿する保険的対策。ブリッジ構成対応後にどこまで必要性が残るか再検討。
- [運用] gnc_pose_relay/PoseRelayClient を control_node に組み込むか: GNC最小構成と組み合わせた場合の効果は未検証（現在停止中）。
- [C] 誤差→力/トルク変換則の差し替え（具体名未定）: 現行はP+D固定則。PID等への差し替えは、候補が具体名で2つ以上出た時点で`docs/arch/architecture_guidelines.md`の昇格ルールに従って判断する。
- 障害物回避中に向きがずれる（オフラインで最大26〜30°、JEMの場面で12.4°）のを許容するか。シムでの計測記録はない（詳細: `docs/archive/achieved/2026-09-24_obstacle_avoidance_local_cost_plan.md`）
- ISSの回転による見かけの力を計画で無視してよいか（ISSの回転は遅く、制御のフィードバックで吸収できると考えているが未検証。詳細: `docs/archive/achieved/2026-09-24_obstacle_avoidance_production_integration_plan.md`）
- `guidance.minco_obstacle_avoidance`（既定`false`）を既定でONにするか
