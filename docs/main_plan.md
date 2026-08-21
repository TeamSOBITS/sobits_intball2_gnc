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

## 未達成タスク（優先度順）

### [G] Guidance軌道のリアルタイム更新（実TFフィードバックによる再計画）の前提条件整備
`look_at`本体実装・90°超waypoint分離機動・経路の補間方式など複数の[G]タスクが実質的に「毎tick実TFを見て参照を更新する」という同じ基盤に帰着する。徹底調査の結果、素朴な実装（毎tick/10Hzで実位置・実速度から再計画）は到着判定のZeno的機能不全・オーバーシュート・目標近傍の飽和/チャタリング・姿勢制御の恒久機能不全など複数の破綻が数学的に確定しており、当初想定より前提条件の整備が先に必要。詳細・破綻の定量的根拠: `docs/guidance_realtime_replanning_design.md`。

着手順（前提条件、依存関係順）:
1. `guidance_node`単一起動の確認手順整備（多重起動が以降の検証結果を無効化するため最初に）
2. `align_at_arrival`の開始条件を時間ベースから位置誤差+settle-timeベースに変更（再計画方式と無関係に単体で価値がある独立改善）
3. Guidance側にTF staleness判定を新設（現状ゼロ）
4. Guidance側に速度推定器を新設
5. `HeuristicSegmentTimeAllocator`にv_now（始点速度）を渡す拡張
6. 残距離下限での再計画停止/静的軌道へのフォールバック方針の決定
7. `Trajectory`内部状態（`_last_q_des`/`_last_sample_t`）の再計画間引き継ぎ設計、`max_angular_rate`を常時指定
8. 再計画レート（Control側P+D 50Hzより明確に低く）のパラメータ設計

1〜3は独立して並行着手可能。4→5→6→7は速度推定→時間配分→フォールバック→状態管理の順で依存。8は実装直前。

### [C] trajectory_controller.kp_att の再設計
移動中の姿勢追従ゲイン。トルク予算0.0047Nm（力優先制約下で確定）を前提に再設計する。詳細: `docs/archive/achieved/session_2026-08-20_thrust_allocator_force_crush_fix.md`

### [G] 90°超waypointでの分離型機動
偏角（進入方向と離脱方向の角度差）が90°を超える経路はゲインチューニングでは解決不可（ファンの物理的トルク上限）。停止→再配向（静止hold機構）→並進再開、の機動にGuidance側で自動的に切り替える設計が必要。90°以下は現行の連続追従（`kp_att=0.60`）で対応可能。詳細: `docs/archive/achieved/trajectory_force_duration_investigation.md`（6-15/6-16節）

### [G] min_snap.py のコアロジック実装（別担当者）
Phase 2で契約は確定済みだがコア実装が未着手。pure functionとして実装（ROS import禁止）。理論: Mellinger & Kumar (2011)。入出力契約: `docs/min_snap_interface_contract.md`。参考実装: https://github.com/The-SS/quadrotor_trajectory 、参考解説: https://dev10110.github.io/tech-notes/control-theory/min_snap.html
成果物: `test_min_snap.py`に数値解検証テスト追加、`test_trajectory_generator_contract.py`の対象に`MinSnapTrajectoryGenerator`追加、`test_segment_time_to_trajectory_pipeline.py`を差し替えて統合確認。依存関係なし（他タスクと並行可）。

### [N] 実機用の自己位置・姿勢推定（IMU単独）
実機にTFは存在しない（TFはシム限定のオラクル）。並進はIMU二重積分で誤差が時間の2乗で発散し長時間精度維持が原理的に困難。姿勢はジャイロ一重積分でより現実的だが、無重力下では加速度計による重力基準補正が使えない。`navigation/utils/`を新設する方針で合意済みだが着手時期・優先度は未定。詳細: `docs/archive/achieved/phase0_findings.md`（観測13）

### [運用] ROS1↔ROS2ブリッジの本番トピック構成方針
GNC最小構成（`/clock`・`/tf`・`/tf_static`・`/imu/imu`・`/gnc/body_pose_raw`等）に絞るだけで`/tf`の負荷耐性が大幅改善することは確証済み。これを正式運用にするか、`bridge_topics.yaml`を用途別複数用意する仕組みにするか、方針を固める必要がある。詳細: `docs/archive/achieved/recording_cpu_load_control_degradation.md`

### [運用] 姿勢のわずかな追従ズレへの対応方針
力優先修正の副作用でトルク予算が約4割減り、姿勢追従に軽微なズレが残る。配分アルゴリズム側の改善余地はほぼ無いと確認済み。「現状維持（ファン物理上限由来として許容）」か「力の精度を意図的に数%犠牲にするトレードオフを試す」かの判断が必要。詳細: `docs/archive/achieved/session_2026-08-20_thrust_allocator_force_crush_fix.md`

### [G] attitude_reference_mode=look_at 本体実装
対象TFフレーム名を`guidance.look_at_target_frame`（`ros2 param`）で受ける設計までは確定済みだが、`_run_trajectory`ループ内で毎tick TFルックアップして姿勢を再計算する本体が未着手。現在選択すると`face_travel`にフォールバックし警告ログのみ出す。TFロスト時のフォールバック方針（直前の`q_des`保持 or goal中断）も未決定。

### [G] pre_align と look_at 併用時の事前整列目標方向
現在の実装は`face_travel`時のみ`v_early`方向に事前整列。`look_at`本体実装時に、事前整列先を「初期タンジェント」から「look-at対象方向」に切り替える設計が必要。

### [G] 移動中のロール継続追従（着陸後の`align_at_arrival`高速化狙い）
`attitude_reference_mode`(`face_travel`/`look_at`)はピッチ・ヨー（進行方向を向く方向）を経路に応じて決めるが、その向きを軸にした回転（ロール）は決めない。移動中ロールが放置されると到着時に大きなロール誤差が残り、`align_at_arrival`の補正が遅くなる（角度が大きいほど遅く・精度も悪化する、`docs/2026-08-21_tf_correction_align_slow_investigation.md`のゲイン実測で確認済み）。移動中もロールを目標値へ継続的に追従させておけば到着時の補正が小角度（速く・高精度）になるはず、という設計アイデア。`trajectory_controller`側でロール成分だけ目標追従させる形の実装が必要。

### [G] 経路の補間方式（直線移動モード）
waypoint間を滑らかに補間するか、ただの直線でつなぐか未検討。姿勢モードとは直交する軌道生成側の話。`BaseTrajectoryGenerator`に3つ目の実装を追加するか、既存`HermiteSplineTrajectoryGenerator`のパラメータで代替できないか検討する。

### [運用] デッドレコニング（安全弁）の実装要否
TF停滞時に最後の推定速度で位置を前方外挿する保険的対策。ブリッジ構成対応後にどこまで必要性が残るか再検討。

### [運用] gnc_pose_relay/PoseRelayClient を control_node に組み込むか
GNC最小構成と組み合わせた場合の効果は未検証（現在停止中）。

### [運用] 一時デバッグ計装・作業ファイルの後片付け
`trajectory_controller.py`内の`/tmp/trajectory_reference_race_timing.log`書き込み（`# TEMPORARY debug instrumentation`で検索）、関連送信スクリプトの一時ログ、`/root/bridge/`配下の未使用ファイル（`bridge_topics_tf.yaml`、`bridge_topics.yaml.bak_*`）の要否判断。消すなら恒久的なデバッグフラグ化も検討可。

### [運用] /ctl/duty のforeign publisher競合の調査
`control_node`のログに`FOREIGN duty messages: fan control is CONTESTED`（`ros_bridge`経由の別publisher）が断続的に出る。JAXA Navigation機能からの独立要件と関連する可能性があり、原因未調査。

### [運用] シム/bridge/gnc_bringup起動順序によるホバー保持不能の再発調査
シム・ROS1↔ROS2ブリッジ・`gnc_bringup.launch.py`の起動順序やタイミングのズレが原因と思われる、ホバー保持ができなくなる現象が複数回再発している（`/ctl/duty`のforeign publisher競合は原因ではないと確認済み）。`control_node`再起動で復帰することは確認済みだが、根本原因（起動順・タイミング依存の何か）は未特定。再現条件の特定と恒久対策が必要。TFのデータや時間が汚染され自己位置が汚染される可能性

### [運用] guidance_nodeの多重起動防止
`ros2 run`で新しい`guidance_node`を起動する際、古いプロセスが`kill`後も子プロセスとして生き残るケースがあり、`/gnc/move_to`に対して同名の`CtlCommandActionServer`が2つ同時に応答する状態が発生した（`Ignoring unexpected goal response. There may be more than one action server`警告、goalの受理・feedbackが混線し検証結果が無効化した）。起動前の既存プロセス検出・確実な停止（プロセスグループ単位でのkill等）か、二重起動自体を検知して片方が終了する仕組みが必要。

### [運用] /gnc/checkpoints送信スクリプトのディスカバリ・レース修正
固定`time.sleep(1.0)`待ちだけでpublishすると購読者マッチングに間に合わずチェックポイントが黙って届かないことがある。`test/manual/send_checkpoints.py`・`send_to_nav_entry.py`が未修正のまま残っている。詳細: `docs/archive/achieved/trajectory_force_duration_investigation.md`（6-7節）

### [Teleop] 手動操縦の最大速度の実測
手動操縦コマンドで機体が到達しうる最大並進/回転速度を実測する。

### [Teleop] 緊急ブレーキ機能の要否・実装方針
手動操縦中に急停止させる機能。`/gnc/stop`検討と合わせて整理する。

### [Teleop] 停止性能の実測
最大速度からの停止までの最長時間・最長距離を実測し、緊急ブレーキ機能や安全マージン設計の根拠にする。

### [運用] 緊急停止/中断サービス（/gnc/stop）の要否（優先度低）
`/gnc/advance_checkpoint`と同じ`std_srvs/Trigger`パターン。Action自体は`cancel_goal`で個別の移動を中断できるため、それと独立した「今すぐ全部止める」手段が本当に必要か未検討。

### [運用] ib2_msgs/FanStatus（duty＋電源状態）の購読（優先度低）
制御ループには不要、診断用途のみの候補。

### [C] 誤差→力/トルク変換則の差し替え（具体名未定）
現行はP+D固定則。PID等への差し替えは、候補が具体名で2つ以上出た時点で`docs/architecture_guidelines.md`の昇格ルールに従って判断する。

### [G] min snap以外の軌道生成代替手法（具体名未定）
具体名はまだ出ていない。

### [将来] 障害物回避（EGO-Planner風、Phase 3完了後）
Minimum Snapが出す軌道が障害物と衝突する場合に局所的に押し出して回避する層を追加する。新設ファイル: `guidance/utils/rebound_optimizer.py`。衝突区間の検出、衝突のない誘導経路の生成、制御点ごとの押し出しベクトル計算、勾配降下（またはscipy等）による再最適化。参考: EGO-Planner（Zhou et al., 2020, RA-L）。
前提として未整理な質問: Guidance側グローバル経路計画（`global_planner/`のA*/RRT）の仕組み確認、Navigationパッケージ構造全般の再確認、障害物情報の取得元（固定マップかセンサーか）、衝突検出→再最適化ループの具体的動作イメージ、障害物マップの保持方式（OctoMap/ESDF/ガウシアンPLY等）。

---

## 未決定事項（判断待ちの論点、タスクではない）

- 実機ではTFが存在しないため、いずれ自前の姿勢推定器（ジャイロ積分＋相補フィルタ等）・自己位置推定器が必要になる。`navigation/utils/`新設の方針で合意済みだが着手時期・優先度は未定（詳細: `docs/archive/achieved/phase0_findings.md`観測13。上記タスク「実機用の自己位置・姿勢推定」と関連）

- `align_at_arrival`の開始条件は**時間ベース**（`_run_trajectory`が軌道の計画所要時間`total_duration`経過で`STATUS_SUCCESS`を返した時点、`guidance_executor.py`）で、実際の位置誤差の大きさは見ていない。並進の追従遅れ・オーバーシュートが残っていてもその瞬間にalign_at_arrivalへ移行し、`PoseCorrector`は位置・姿勢を同じcheckpointで**同時に**補正する（今も既に「並進修正しながら回転」）。並進誤差が大きいまま同時補正に入ると、`thrust_allocator`の力優先重み付け（`docs/archive/achieved/session_2026-08-20_thrust_allocator_force_crush_fix.md`）で回転トルクの予算が削られ、回転が遅くなる可能性がある。「並進を先に収束させてから回転する（順番実行）」方が、並進誤差が大きいケースでは総時間で速くなり得るが、現時点（2026-08-21）で検証した3ケースでは同時実行のままでも暴れずに収束しており、対策の必要性は未確認（`docs/archive/achieved/2026-08-21_align_torque_budget_crush_hypothesis.md`）。大きな並進誤差を伴う経路でalign_at_arrivalが遅い/暴れるケースが実際に出たら、位置誤差が閾値以下になるまで姿勢補正を保留する分岐を検討する。
