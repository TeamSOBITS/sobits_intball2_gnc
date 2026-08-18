# navigation （Navigation: GNCの「N」）

**現状のスコープ**: 「名前付き地点をTFとして配信する」機能のみです。狭義の自己位置推定（自機の位置・姿勢をセンサーから推定すること）は範囲外で、自機位置はシミュレータのTF（`iss_body`<-`body`）が真値として配信するものをそのまま利用しています（`control/`側が消費）。

```
navigation/
├── location_broadcaster.py  # maps/iss_location.yaml の登録地点を TF として配信（10Hz）
└── location_setting.py      # ロケーション登録 GUI（Tkinter + Zenity）
```

## 地点登録方法

1. YAML に登録された地点を TF としてパブリッシュする
   ```sh
   ros2 run sobits_intball2_gnc location_broadcaster
   ```
   - `maps/iss_location.yaml` の地点を `iss_body` 座標系の TF フレームとして 10Hz で配信します
   - YAML ファイルの変更をリアルタイムで検知し，再起動なしに反映します

2. ロケーション登録 GUI を起動する
   ```sh
   ros2 run sobits_intball2_gnc location_setting
   ```
   - Zenity のファイル選択ダイアログが開きます（デフォルト: [maps/iss_location.yaml](../../maps/iss_location.yaml)）
   - シミュレータ内でロボットを登録したい地点・姿勢に移動させ，GUI で地点名を入力して **SNAP CURRENT POS** を押すと登録されます
   - 登録済みの地点の削除・リネームも GUI から行えます

## 既知の制約（実機化に向けて）

このパッケージの自己位置は，シミュレータが配信するTF（`iss_body`<-`body`）に依存しています。これは**シミュレータ限定の近似真値**で，実機には存在しません。実機でIMU単独（他センサーなし）で自己位置・姿勢を求める場合，以下の原理的な制約に注意が必要です。

- **並進位置の推定**（加速度計の二重積分）は，誤差が時間の**2乗**で発散するため，外部参照（カメラ・マーカー・超音波等）なしでの長時間の精度維持は現実的に困難です。
- **姿勢の推定**（ジャイロの一重積分）は誤差が時間に**比例**するため相対的に現実的ですが，無重力環境（ISS内）では加速度計による重力方向の検知ができず，地上のロボットで一般的な「重力を基準にした傾き補正」が使えません。

将来，自前の姿勢推定器・自己位置推定器（相補フィルタ／Kalmanフィルタ等）を実装する場合は，このディレクトリに `navigation/utils/`（ROS非依存のpure functionレイヤー）を新設し，`control/utils/` と同じ設計方針（素値コンストラクタ・単体テスト可能）を踏襲することを想定しています。
