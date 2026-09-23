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

### [G] MINCO v4：実際の速度推定を入れると破綻（根本原因未特定、Phase4着手のブロッカー）
「完璧な速度」前提では<1mm収束するが、実際のVelocityEstimator相当（有限差分＋EMA）を入れると18〜99mに破綻する。トランジェント要因・tick0初期化バグは既に切り分け済みで否定されている。原因が判明するまで本番実装（Phase4）には進まない方針。詳細: `docs/archive/2026-09-01_replanning_minco_v4_open_issues.md`

### [G] 中断時（cancel）の制動プロファイル追加
巡航中0.5m/sでcancelすると即座に現在位置を保持目標にするため、PDが止めるしかなく2m以上オーバーシュートしうる。JAXA型の停止距離逆算プロファイルで解消。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（A-3）

### [G] 実行不能時フォールバック順序の見直し（Hermite台形をウレンチ制約無視のまま最終手段にしない）
現状の最終フォールバック（Hermite台形）はウレンチ包絡域を一切考慮しておらず、約92%のサンプルで真の到達可能領域を超える危険側の設計。TOPP-RA/MINCO→逐次実行（並進→回転）→直線+単軸回転、の順に変更する。詳細: 同上（B-1）

### [G] 微小移動指令のデッドゾーン
5cm未満のような微小指令でもA*→Hermite→TOPP-RAのフルパイプラインが走り、`static_minco`時は3〜5秒ブロックする。閾値未満は即`STATUS_SUCCEEDED`で返す。詳細: 同上（B-2）

### [G] K分離(2自由度版)、短距離レグでのduration悪化が未改良
MINCOの姿勢waypoint密度↑時のduration悪化は圧縮できた（+133%→+21%等）が、短距離レグでは悪化が
残ったまま（`docs/archive/2026-09-01_replan_speedup_options_overview.md`）。要改良。

### [G] attitude_reference_mode=look_at 本体実装
対象TFフレーム名を`guidance.look_at_target_frame`（`ros2 param`）で受ける設計までは確定済みだが、`_run_trajectory`ループ内で毎tick TFルックアップして姿勢を再計算する本体が未着手。現在選択すると`face_travel`にフォールバックし警告ログのみ出す。TFロスト時のフォールバック方針（直前の`q_des`保持 or goal中断）も未決定。

### [G] pre_align と look_at 併用時の事前整列目標方向
現在の実装は`face_travel`時のみ`v_early`方向に事前整列。`look_at`本体実装時に、事前整列先を「初期タンジェント」から「look-at対象方向」に切り替える設計が必要。

### [G] min snap以外の軌道生成代替手法（具体名未定）
具体名はまだ出ていない。

### [G] 移動前のロール事前回転（到着後の`align_at_arrival`高速化狙い）
`attitude_reference_mode`(`face_travel`/`look_at`)はピッチ・ヨー（進行方向を向く方向）を経路に応じて決めるが、その向きを軸にした回転（ロール）は決めない。移動中ロールが放置されると到着時に大きなロール誤差が残り、`align_at_arrival`の補正が遅くなる（角度が大きいほど遅く・精度も悪化する、`docs/archive/achieved/2026-08-21_tf_correction_align_slow_investigation.md`のゲイン実測で確認済み）。移動中にロールも回転させると貴重な推力が減ってしまう。そのため、移動前に、到着後のロールだけでも合わせておくことで、事後回転の高速化が狙えるはず。優先度中

### [G] 経路の補間方式（直線移動モード）
waypoint間を滑らかに補間するか、ただの直線でつなぐか未検討。姿勢モードとは直交する軌道生成側の話。`BaseTrajectoryGenerator`に3つ目の実装を追加するか、既存`HermiteSplineTrajectoryGenerator`のパラメータで代替できないか検討する。

### [G] wrench_envelope_safety_marginのさらなる調整余地
`guidance.wrench_envelope_safety_margin=0.7`＋`thrust_allocator.minimax_objective=true`で
duty≥0.95飽和頻度を48%→33.4%まで改善したが、まだ33%残っている（`docs/archive/achieved/
2026-08-28_toppra_static_path_attitude_overshoot_incident.md`その7）。安全係数をさらに下げる
（0.6/0.5）ことで飽和頻度と所要時間のトレードオフを追加探索する余地がある。

### [G] minco_wrench_safety_marginの既定値見直し（TOPP-RA側と統一）
TOPP-RA側は`wrench_envelope_safety_margin=0.7`だが、MINCO側は既定`1.0`（FB余力なし）で攻めすぎ。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（C-2）

### [G] 慣性テンソルの非等方性検証
等方前提だとジャイロ項ω×(Iω)がゼロになり無視できるが、実機の慣性テンソルが非等方ならこの項が復活し角速度の2乗で効いてくる。現状の等方スカラー0.0136 kg·m²が実機と一致するか未検証。

### [G] 抗力トルク実測値（κ_j/k_j）をAに反映して包絡を再構築（シム未対応のため実機フィデリティはシム上で検証不可）
三谷・西下・平野2023の実測値から8基分のκ_j/k_jは同定済み・番号対応も検証済み（ロールで36%の能力回復、力とヨーは正しく1〜2割補正、面数24→112）。実装対象は自分たちの`A`行列（`actuation_envelope.py`）のみでシム変更は不要だが、現在のシムは`kappa=0`（抗力トルクなし）のままなので、シム上ではこの改善による実機フィデリティ向上分を検証しようがない点に注意。付随課題として実機の推力方向ベクトル未入手（公称モデルとの乖離あり、Y方向が実測で3割弱い；著者への問い合わせ候補）。

### [G] min_snap.py のコアロジック実装（別担当者、優先度低）
Phase 2で契約は確定済みだがコア実装が未着手。pure functionとして実装（ROS import禁止）。理論: Mellinger & Kumar (2011)。入出力契約: `docs/minimum_snap/min_snap_interface_contract.md`。参考実装: https://github.com/The-SS/quadrotor_trajectory 、参考解説: https://dev10110.github.io/tech-notes/control-theory/min_snap.html
成果物: `test_min_snap.py`に数値解検証テスト追加、`test_trajectory_generator_contract.py`の対象に`MinSnapTrajectoryGenerator`追加、`test_segment_time_to_trajectory_pipeline.py`を差し替えて統合確認。依存関係なし（他タスクと並行可）。`static`モードは既にTOPP-RA（`ToppraTrajectory`）で力/トルク制約を考慮した軌道生成に置き換わっているため、緊急度は下がっている。

### [C] 姿勢制御へのフィードフォワード導入（ω_des/α_des）とカスケード化
姿勢制御が純粋PDでFFが無く、軌道側で計算済みの`vel[3:]`/`acc[3:]`（回転ベクトルの1・2階微分）を捨てたまま。`sample()`の戻り値拡張と、大回転時のJacobian補正（回転ベクトル微分→角速度変換）が必要。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（A-1）

### [C] omega_errを数値微分から解析値へ置き換え
現状50Hzの数値微分（`att_filter_alpha=1.0`で無フィルタ）でノイズと位相遅れを抱えている。上記A-1で`w_des`が得られれば`omega_imu - R(qe)*w_des`で解析的に算出できる。A-1とセットで実施。詳細: 同上（A-2）

### [C] thrust_allocatorの決定的な配分フォールバック
`lsq_linear`（反復ソルバ、計算時間が非決定的）に時間上限を設け、超過・失敗時はJAXA型の行列配分（事前計算行列の積と最小値減算のみ、固定時間）に落とす。詳細: 同上（B-4）

### [C] 追従誤差ガード（`Dtc`相当）の追加
追従誤差が閾値を超えても軌道を中断しない。特に`static`モードは開ループのため機体位置に関わらず基準時刻が進み続け、外乱・衝突を検知できない。閾値超過で中断時制動プロファイル（[G]中断時の制動プロファイル追加）経由のholdへ落とすガードが必要。詳細: 同上（C-1）

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

### [C] 積分項の追加（実験・優先度低）
JAXAは`ki`と飽和付き積分器（`fi_max=0.02`）が配線済み（現状`ki=0`）だが、SOBITSには無い。重心オフセット・配分行列誤差由来の定常バイアス切り分け材料として、`ki=0`から始めればリスクなく実験できる。詳細: `docs/arch/2026-09-20_jaxa_to_sobits_backport_candidates.md`（B-3）

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
実装方針の代替案: 新規Pythonファイルより`minco_solver.cpp`内に衝突ペナルティ項を足す方が工数小。積分ノード評価ループ・smoothed-L1ペナルティ・勾配蓄積・重みスケジュール・L-BFGSを再利用でき、`viol = F_ENV·w - g`を`viol = d_safe - d(p)`に置き換えるだけで構造が同じ。レンチ制約と衝突制約が同じ最適化内で同時に効く。前提としてESDF（距離場）が必須（勾配が要るためOctoMap単体では不可）。

### [将来] MPCC姿勢/トルク統合の実行可能性課題
並進のみのMPCCはprogress stall解決済み・強擾乱250〜1000tickでinfeasible/予算超過ゼロを確認済み。しかし姿勢/トルク統合プロトタイプでは、弱擾乱時にACADOS_MINSTEPで解が不可解になる、強擾乱時は並進と姿勢が8ファンの推力予算を奪い合い並進収束が4mm→407mmへ悪化する、という新課題が判明。ソルバ時間も13〜15ms/tickに増加（100ms予算内ではある）。本番導入するか自体が未定。詳細: `docs/archive/2026-08-29_mpcc_attitude_torque_integration_plan.md`

### [将来] 並進・姿勢分離の妥当性を数値で正当化
Watterson, Smith & Kumar (IROS 2016)は「全軸の推力能力が同程度なら分離が妥当」と明言しているが、Int-Ball2の包絡は方向によって約58%の開き（`|b|`が0.002516/0.002835/0.003969の3種類）があり、この前提が成立していない可能性がある。定量化できれば位置と姿勢を統合的に最適化する設計判断の根拠になる（論文化する場合は新規性の主張の中核にもなり得る）。

### [将来] 急カーブでの定量比較データ取得（論文用、動作可否は確認済み）
TOPP-RA staticでどんな急カーブでも曲がれること自体は確認済みで、これは動作可否の課題ではない。残るのはIAC-22（振幅0.4m・長さ6.0m・ウェイポイント間隔0.5m）と同条件から曲率をきつくしていき、先読み距離方式が破綻する点との対比データを取る作業で、論文執筆時にのみ必要。ISS船内でRRT*が出す回避経路が実際にどの程度の曲率を要求するかは別途確認が要る。

---

## 未決定事項（判断待ちの論点、タスクではない）

- 実機ではTFが存在しないため、いずれ自前の姿勢推定器（ジャイロ積分＋相補フィルタ等）・自己位置推定器が必要になる。`navigation/utils/`新設の方針で合意済みだが着手時期・優先度は未定（詳細: `docs/archive/achieved/2026-08-19_phase0_findings.md`観測13。上記タスク「実機用の自己位置・姿勢推定」と関連）
