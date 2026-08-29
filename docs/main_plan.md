# 対象機体（IntBall2）に即した実装計画 — 未達成タスク

**対象**: `sobits_intball2_gnc`（ISS内自由飛行キューブ型ロボット IntBall2 の GNC 実装、ROS2/Humble）

**機体特性**: 無重力環境、8ファンによる全方向駆動（fully-actuated）→ 並進と姿勢を独立に制御できる

## このファイルの書き方（方針）

- **未達成タスクのみ**を書く。達成済みの内容は書かない。達成済みの経緯・調査結果は`docs/archive/achieved/`配下を参照する。
- **Phase番号は使わない**。フラットな優先度順リストとする（上にあるものほど優先度が高い）。
- 各タスクは `[タグ] タスク名` の見出し＋短い動機（1文、目安40字程度）＋必要なら参照リンク、の形式で書く。
  - タグはGNCの役割（G/N/C）または領域（運用/Teleop/将来）を示す目印。分類のためではなく一目で系統がわかるようにするための軽い印。
  - 動機は「なぜ今これが必要か」のみ。経緯・調査の数値的根拠・比較実験の詳細はarchiveに委ねる。
  - **例外**: 未達成の判断そのものに直結する数値・閾値（例: トルク予算の値、角度の閾値）は動機に残してよい。それが無いとタスクの意味が読めなくなるため。
- **テーブル形式は使わない**（スキャン性より、経緯を含む短い文章の方がこのファイルには合う）。
- 「タスク」と「未決定事項（判断待ちの論点でタスクではないもの）」は別セクションに分ける。混ぜない。

---

## 未達成タスク（カテゴリ別・各カテゴリ内は優先度順）

### [G] replanningモードが力/トルク制約を考慮しない設計のまま
`static`モードは`ToppraTrajectory`＋実`ThrustAllocator`ベースの厳密なウレンチ包絡域制約
（`guidance/utils/actuation_envelope.py`）に置き換え済み（`docs/archive/achieved/
2026-08-28_toppra_static_path_attitude_overshoot_incident.md`）だが、`replanning`モードは
今も`HeuristicSegmentTimeAllocator`の単一スカラー`max_accel`＋`HermiteSplineTrajectoryGenerator`
（力/トルク非参照、幾何的にのみ軌道を作る）のままで、生成された軌道が各瞬間・各軸で実現可能かを
検証する層が無い。`replanning`はTOPP-RA化しない設計方針（`sd_start`がスカラー速度のみで
`v_perp`を扱えないため）のため、この構造的欠陥は別の手当てが必要。上記のsegment_time_infeasible
バグと同根の可能性がある。また、`static`モードで見つかった大角度旋回時の同種の問題
（姿勢オーバーシュート・並進停滞）が`replanning`モードでも起きないか、別途sim検証で確認が必要
（既存のrate-limit機構があるため挙動が異なる可能性が高いが未検証）。

### [G] MINCO/GCOPTERベースの姿勢/トルク統合軌道生成（速度改善待ち、上記replanning課題の候補解）
将来の障害物回避は並進経路が急に曲がることを前提にしており（下記[将来]障害物回避タスク）、その
急旋回に`face_travel`が幾何のみで追従しようとすると`static`モードで一度踏んだ姿勢オーバーシュートを
`replanning`モードでも踏む懸念がある（`main_plan.md`との照合済み、`docs/
2026-08-29_minco_attitude_torque_integration_plan.md`「軌道再生性」項目）。この解決候補として、
位置3軸+回転ベクトル3軸を1本の6自由度MINCO軌道として扱い、実`wrench_envelope_halfspaces`を
ペナルティとして統合するプロトタイプ（GCOPTER公式C++実装ベース）を試作し、数値的には健全
（並進のみ試作5とのクロスチェック、feasibility確認済み）だが、**求解時間が並進のみ試作5
（1.5〜1.9ms）の約450〜560倍（0.83〜0.93秒）**——`static`相当の一発計画には十分実用的だが
`replanning`の10Hz予算には全く届かないと判明。ボトルネックは実envelopeの9951面評価コストで、
高速化3案（ベクトル化/2段階足切り/面数削減、後者2つは正しさとのトレードオフあり）を整理済み・
未着手。詳細: `docs/2026-08-29_minco_attitude_torque_integration_plan.md`、
`docs/2026-08-29_minco_gcopter_survey.md`。

### [G] replanningモードでのsegment_time_infeasible時の大きなオーバーシュート
`trajectory_tracking_mode=replanning`で、残距離方向と実速度の横方向成分（v_perp）が
大きくずれる状況（例: 距離約4.7mの斜め方向move_to）で`HeuristicSegmentTimeAllocator`の
`t_min_perp > t_max`が成立し`segment_time_infeasible`で再計画が完全停止、その後は
`GuidanceExecutor`の`max_accel`のみに頼る弱い収束になり、目標を約1m追い越してから
事後のcheckpoint holdで収束する、という挙動を実sim検証で確認（2026-08-27、
`docs/archive/achieved/2026-08-27_max_force_anisotropy_from_fan_model.md`検証結果参照）。
`docs/2026-08-25_v0_aware_time_allocation_lateral_velocity_fix.md`が同じ失敗モードを
既に予言・記録していたが、本番規模のシナリオで実際に発生を確認したのは今回が初めて。
再計画停止後のフォールバック挙動（オーバーシュートを許容せず縮退する等）の見直しが必要。

### [G] attitude_reference_mode=look_at 本体実装
対象TFフレーム名を`guidance.look_at_target_frame`（`ros2 param`）で受ける設計までは確定済みだが、`_run_trajectory`ループ内で毎tick TFルックアップして姿勢を再計算する本体が未着手。現在選択すると`face_travel`にフォールバックし警告ログのみ出す。TFロスト時のフォールバック方針（直前の`q_des`保持 or goal中断）も未決定。

### [G] pre_align と look_at 併用時の事前整列目標方向
現在の実装は`face_travel`時のみ`v_early`方向に事前整列。`look_at`本体実装時に、事前整列先を「初期タンジェント」から「look-at対象方向」に切り替える設計が必要。

### [G] min snap以外の軌道生成代替手法（具体名未定）
具体名はまだ出ていない。

### [G] 90°超waypointでの分離型機動（優先度低、TOPP-RA導入後は不要の可能性が高い）
旧`compute_q_des`連続追従・経路全体で単一の`T`固定・`kp_att=0.60`という設計を前提に、偏角90°超は
ゲインチューニングでは解決不可と判定していた（`docs/archive/achieved/
2026-08-19_trajectory_force_duration_investigation.md`6-15〜6-18節、144.7°hairpinで`T`を100秒に
伸ばし`kp_att`を倍にしても131°ピークの遅れが解消しなかった）。その後導入したTOPP-RA（`static`
モード）で実在waypointによる143.99°の鋭旋回（`above_dock`→`inspection_entry_3`経由→
`capture_point_1`）を再検証したところ、action SUCCESS・duty≥0.95飽和25.7%（`near_dock`系ルートの
33.4%より低い）・飛行中の姿勢誤差最大18.1°・最終到達精度も問題なしという結果が得られ、旧分析が
懸念した破綻は再現しなかった（`docs/archive/achieved/
2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その11）。分離型機動は**不要になった
可能性が高い**が、この1ケースのみの確認のため、複数の異なる鋭旋回ケースで再現性を確認してから
このタスク自体を削除するのが安全。

### [G] 移動前のロール事前回転（到着後の`align_at_arrival`高速化狙い）
`attitude_reference_mode`(`face_travel`/`look_at`)はピッチ・ヨー（進行方向を向く方向）を経路に応じて決めるが、その向きを軸にした回転（ロール）は決めない。移動中ロールが放置されると到着時に大きなロール誤差が残り、`align_at_arrival`の補正が遅くなる（角度が大きいほど遅く・精度も悪化する、`docs/2026-08-21_tf_correction_align_slow_investigation.md`のゲイン実測で確認済み）。移動中にロールも回転させると貴重な推力が減ってしまう。そのため、移動前に、到着後のロールだけでも合わせておくことで、事後回転の高速化が狙えるはず。優先度中

### [G] 経路の補間方式（直線移動モード）
waypoint間を滑らかに補間するか、ただの直線でつなぐか未検討。姿勢モードとは直交する軌道生成側の話。`BaseTrajectoryGenerator`に3つ目の実装を追加するか、既存`HermiteSplineTrajectoryGenerator`のパラメータで代替できないか検討する。

### [G] test_toppra_trajectory.py に大角度旋回の回帰テストが無い
`static`モードで実際に発生した大角度旋回（waypoint間で90°近い方向転換）による姿勢オーバー
シュート・並進停滞事案（`docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`）は修正済みだが、この規模の旋回を再現する回帰テストケースが
`test_toppra_trajectory.py`に無い。同種の再発を検知できるよう追加が必要。

### [G] wrench_envelope_safety_marginのさらなる調整余地
`guidance.wrench_envelope_safety_margin=0.7`＋`thrust_allocator.minimax_objective=true`で
duty≥0.95飽和頻度を48%→33.4%まで改善したが、まだ33%残っている（`docs/archive/achieved/
2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その7）。安全係数をさらに下げる
（0.6/0.5）ことで飽和頻度と所要時間のトレードオフを追加探索する余地がある。

### [G] min_snap.py のコアロジック実装（別担当者、優先度低）
Phase 2で契約は確定済みだがコア実装が未着手。pure functionとして実装（ROS import禁止）。理論: Mellinger & Kumar (2011)。入出力契約: `docs/min_snap_interface_contract.md`。参考実装: https://github.com/The-SS/quadrotor_trajectory 、参考解説: https://dev10110.github.io/tech-notes/control-theory/min_snap.html
成果物: `test_min_snap.py`に数値解検証テスト追加、`test_trajectory_generator_contract.py`の対象に`MinSnapTrajectoryGenerator`追加、`test_segment_time_to_trajectory_pipeline.py`を差し替えて統合確認。依存関係なし（他タスクと並行可）。`static`モードは既にTOPP-RA（`ToppraTrajectory`）で力/トルク制約を考慮した軌道生成に置き換わっているため、緊急度は下がっている。

### [C] trajectory_controller のTF速度推定ノイズ調査
move_to中に`f_des`が瞬間的に0.68N超まで跳ねる事象を観測、計画側（`a_des`/`v_des`）はほぼ無風
だったためフィードバック起因（`kd_pos`側）と特定済みだが、`trajectory_controller`のTF速度推定
（差分→`vel_filter_alpha`のEMA）自体のノイズ・遅延特性の単体調査は未着手
（`docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その8）。

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

### [C] 誤差→力/トルク変換則の差し替え（具体名未定）
現行はP+D固定則。PID等への差し替えは、候補が具体名で2つ以上出た時点で`docs/architecture_guidelines.md`の昇格ルールに従って判断する。

### [N] 実機用の自己位置・姿勢推定（IMU単独）
実機にTFは存在しない（TFはシム限定のオラクル）。並進はIMU二重積分で誤差が時間の2乗で発散し長時間精度維持が原理的に困難。姿勢はジャイロ一重積分でより現実的だが、無重力下では加速度計による重力基準補正が使えない。`navigation/utils/`を新設する方針で合意済みだが着手時期・優先度は未定。詳細: `docs/archive/achieved/2026-08-19_phase0_findings.md`（観測13）

### [運用] guidance.via_waypoint がノード再起動でリセットされる
`gnc_params.yaml`に書かれておらず、`ros2 param set`のみでランタイム設定される値のため、
guidance_node再起動のたびに空文字列（経由なし）へリセットされる。特定の経由地点を前提にした
検証を続ける場合はyamlに明記するか、再起動後に`ros2 param get`で確認する運用を徹底する必要が
ある（`docs/archive/achieved/2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その7で
これに気づかず1回検証データを無効にした実例あり）。

### [運用] シム/bridge/gnc_bringup起動順序によるホバー保持不能の再発調査
シム・ROS1↔ROS2ブリッジ・`gnc_bringup.launch.py`の起動順序やタイミングのズレが原因と思われる、ホバー保持ができなくなる現象が複数回再発している（`/ctl/duty`のforeign publisher競合は原因ではないと確認済み）。`control_node`再起動で復帰することは確認済みだが、根本原因（起動順・タイミング依存の何か）は未特定。再現条件の特定と恒久対策が必要。TFのデータや時間が汚染され自己位置が汚染される可能性。シム起動→bridge起動→ホバー制御（`hover_control.launch.py`）起動、のタイミングが早すぎる（TF/センサーデータが安定する前に`control_node`が動き出す）と、機体が急速旋回し続ける現象を確認。再現条件の有力候補: `control_node`（ホバー）起動済みの状態でシム・bridgeを再起動すると発生する。各起動ステップ間に十分な待機・データ安定確認を挟む運用ルールが必要。

### [運用] test_hover_controller.py が12件失敗中
`_FakeAllocator`テストダブルに`achieved_wrench`属性が無く`AttributeError`で失敗。既存の
未完了作業由来（今回のTOPP-RA/duty飽和対策とは無関係）、テストダブルへのメソッド追加で修正見込み。

### [運用] move_to検証時に位置追従誤差（TF vs 計画）も継続計測する
`test/manual/measure_position_tracking_error.py`で計測した位置追従誤差（平均/最大/最終値）は
今回はじめて記録した指標で比較対象となる過去データが無い。yオーバーシュートのような単一軸の
最大逸脱値だけでなく、今後の検証でも複数軸・複数統計量で継続記録する運用にする。

### [運用] ROS1↔ROS2ブリッジの本番トピック構成方針
GNC最小構成（`/clock`・`/tf`・`/tf_static`・`/imu/imu`・`/gnc/body_pose_raw`等）に絞るだけで`/tf`の負荷耐性が大幅改善することは確証済み。これを正式運用にするか、`bridge_topics.yaml`を用途別複数用意する仕組みにするか、方針を固める必要がある。詳細: `docs/archive/achieved/2026-08-19_recording_cpu_load_control_degradation.md`

### [運用] 姿勢のわずかな追従ズレへの対応方針
力優先修正の副作用でトルク予算が約4割減り、姿勢追従に軽微なズレが残る。配分アルゴリズム側の改善余地はほぼ無いと確認済み。「現状維持（ファン物理上限由来として許容）」か「力の精度を意図的に数%犠牲にするトレードオフを試す」かの判断が必要。詳細: `docs/archive/achieved/2026-08-20_thrust_allocator_force_crush_fix.md`

### [運用] デッドレコニング（安全弁）の実装要否
TF停滞時に最後の推定速度で位置を前方外挿する保険的対策。ブリッジ構成対応後にどこまで必要性が残るか再検討。

### [運用] gnc_pose_relay/PoseRelayClient を control_node に組み込むか
GNC最小構成と組み合わせた場合の効果は未検証（現在停止中）。

### [運用] 一時デバッグ計装・作業ファイルの後片付け
`trajectory_controller.py`内の`/tmp/trajectory_reference_race_timing.log`書き込み（`# TEMPORARY debug instrumentation`で検索）、関連送信スクリプトの一時ログ、`/root/bridge/`配下の未使用ファイル（`bridge_topics_tf.yaml`、`bridge_topics.yaml.bak_*`）の要否判断。消すなら恒久的なデバッグフラグ化も検討可。

### [運用] /gnc/checkpoints送信スクリプトのディスカバリ・レース修正
固定`time.sleep(1.0)`待ちだけでpublishすると購読者マッチングに間に合わずチェックポイントが黙って届かないことがある。`test/manual/send_checkpoints.py`・`send_to_nav_entry.py`が未修正のまま残っている。詳細: `docs/archive/achieved/2026-08-19_trajectory_force_duration_investigation.md`（6-7節）

### [運用] 緊急停止/中断サービス（/gnc/stop）の要否（優先度低）
`/gnc/advance_checkpoint`と同じ`std_srvs/Trigger`パターン。Action自体は`cancel_goal`で個別の移動を中断できるため、それと独立した「今すぐ全部止める」手段が本当に必要か未検討。

### [Teleop] 手動操縦の最大速度の実測
手動操縦コマンドで機体が到達しうる最大並進/回転速度を実測する。

### [Teleop] 緊急ブレーキ機能の要否・実装方針
手動操縦中に急停止させる機能。`/gnc/stop`検討と合わせて整理する。

### [Teleop] 停止性能の実測
最大速度からの停止までの最長時間・最長距離を実測し、緊急ブレーキ機能や安全マージン設計の根拠にする。

### [将来] 障害物回避（EGO-Planner風）
Minimum Snapが出す軌道が障害物と衝突する場合に局所的に押し出して回避する層を追加する。新設ファイル: `guidance/utils/rebound_optimizer.py`。衝突区間の検出、衝突のない誘導経路の生成、制御点ごとの押し出しベクトル計算、勾配降下（またはscipy等）による再最適化。参考: EGO-Planner（Zhou et al., 2020, RA-L）。
前提として未整理な質問: Guidance側グローバル経路計画（`global_planner/`のA*/RRT）の仕組み確認、Navigationパッケージ構造全般の再確認、障害物情報の取得元（固定マップかセンサーか）、衝突検出→再最適化ループの具体的動作イメージ、障害物マップの保持方式（OctoMap/ESDF/ガウシアンPLY等）。

---

## 未決定事項（判断待ちの論点、タスクではない）

- 実機ではTFが存在しないため、いずれ自前の姿勢推定器（ジャイロ積分＋相補フィルタ等）・自己位置推定器が必要になる。`navigation/utils/`新設の方針で合意済みだが着手時期・優先度は未定（詳細: `docs/archive/achieved/2026-08-19_phase0_findings.md`観測13。上記タスク「実機用の自己位置・姿勢推定」と関連）
