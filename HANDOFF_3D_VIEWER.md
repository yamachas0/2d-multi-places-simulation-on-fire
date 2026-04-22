# 3D Viewer 作業 引き継ぎメモ

**作成日**: 2026-04-21
**対象**: PC再起動後のセッション復帰用
**前提**: 前会話 (要約あり) + 本会話でshinagawa対応まで完了

---

## 1. これまでの完了作業

### 1-1. Phase 3 プロダクション run 用の修正 (完了)
run `simulations/2026-04-20_2147_13_jr_disruption/` で以下を検証済み:
- 方針D（1:1 partner制）により重複メッセージ率 75.8% → 0.0%
- 時間整合性（朝語彙 0 / 夕方 お疲れ 17件 自然分布）
- 思考ログ常時表示 + クリック追従UI

レポート:
- `simulations/.../V2_DUPLICATE_MESSAGE_FIX.md`
- `simulations/.../V2_TIME_CONSISTENCY.md`
- `simulations/.../V2_THOUGHT_LOG.md`

### 1-2. viewer_v2.html ビジュアルチューニング (完了)
- 赤系夕方背景 → ダーク緑pine (`#0e1714`/`#0d1f1a`→`#1b3426`)
- ステップ間補間 (`state.stepFrac` + `lerpAgentPos` + teleport guard)
- baseStepMs 400→900ms で滑らか化

### 1-3. Git管理 (完了)
- `4c65215 feat(stage-1.5): Phase 2.6 awareness + strategy D + evening prompts`
- `a178c6d feat(viewer): Canvas viewer v2 + analysis tools + figure polish`
- `origin/feat/stage-1.5-urban-scale` に push済

### 1-4. 3D Viewer Phase 1 実装 (完了)
viewer_3d_spec.md に従って `visualization/viewer_3d.html` (~860行) を新規作成。
- Three.js r128 UMD + OrbitControls
- Place: BoxGeometry の1F着色スラブ + 上階ワイヤーフレーム
- Agent: pre-allocated BoxGeometry プール + AwarenessRing (notification/direct/conversation色)
- 会話線: LineDashedMaterial
- イベント: RingGeometry パルスアニメ (sin波)
- Raycaster でクリック選択 + selection halo
- カメラ: 視点リセット/俯瞰/追従
- 右ペイン: 会話ログ / 思考ログ / フォーカス詳細 (viewer_v2から移植)

### 1-5. bundle_viewer.py 拡張 (完了)
- `--viewer {2d,3d}` オプション追加
- 3D時はThree.js UMDを unpkg からfetch→`visualization/vendor/three-0.128/` にキャッシュ→インライン化
- オフライン完結HTMLを出力

### 1-6. Shinagawa config対応 (完了)
`D:\ユーザー\ダウンロード\shinagawa_config.yaml` に対応:

**place type 7種追加** (viewer_3d.html `PLACE_STYLE`):
- hotel `#c8a464`
- residential `#89c0a8`
- industrial `#6b7280`
- utility `#7a8a75`
- construction `#ffb84a`
- zone_izakaya `#d85c7a`
- museum `#b49b5e`

**`attributes.height_m` オーバーライド対応**: place定義側で高さ明示指定できる

**scene_3d 描画** (viewer-only、sim engineは触らない):
- `buildRails` - 1F/2F高架対応、柱レンダ (京急本線)
- `buildRoads` - arterial/collector/local で色分け
- `buildDecks` - elevation 7m のコンクリスラブ
- `buildStairs` - 地上↔デッキ接続の傾斜
- `buildLandmarks` - CanvasTexture スプライトでラベル

**bundle_viewer.py**: `--scene-3d <yaml>` オプション追加。config の非標準 `; 区切り` shorthand を正規化してPyYAML解析、scene_3d subtree を JSON化して `<script id="scene-3d">` に注入。

**動作確認済**:
```bash
python tools/bundle_viewer.py simulations/2026-04-20_2147_13_jr_disruption \\
    --viewer 3d \\
    --scene-3d 'D:\\ユーザー\\ダウンロード\\shinagawa_config.yaml'
# 出力: viewer_3d_bundled.html (1.67MB)
# scene_3d injected: rails 3 / roads 20 / decks 2 / stairs 1 / landmarks 3
```

---

## 2. 生成物 (主要)

```
visualization/
├── viewer_v2.html              2D ビューア (既存、緑系テーマに更新済)
├── viewer_3d.html              3D ビューア (新規、~860行+shinagawa拡張)
└── vendor/three-0.128/         Three.js UMDキャッシュ (.gitignored)

tools/
└── bundle_viewer.py            --viewer {2d,3d} + --scene-3d <yaml> 対応

simulations/2026-04-20_2147_13_jr_disruption/
├── simulation_data.json        run 13 データ (40 personas / 19 places / 60 steps)
├── viewer_v2_bundled.html      2Dバンドル (1.16MB)
├── viewer_3d_bundled.html      3Dバンドル (1.67MB、scene_3dオーバーレイ付き)
└── V2_*.md                     3つのレポート

PROJECT_OVERVIEW_FOR_3D.md      3D化検討用プロジェクト概要 (他AI投入用)
```

---

## 3. 残タスク / 次やること

### 3-1. shinagawa で sim 実行 (未着手)
現状の `viewer_3d_bundled.html` は run 13 (JR disruption / half_space=75 / 19 places) の上に shinagawa scene_3d をオーバーレイしただけ。scene_3d は構造物として描画されるが、sim の place 定義は JR disruption のまま。

**やるべき**: `shinagawa_config.yaml` でシミュを実行して本物のshinagawa runを作る。
- `half_space_size: 80`
- 40+ places (hotels, residential, industrial...)
- personas は既存 config から流用予定 (config側コメントに記載)

```bash
python main.py --config D:\ユーザー\ダウンロード\shinagawa_config.yaml \\
    --output-name shinagawa_rush_01
# → simulations/2026-04-21_HHMM_NN_shinagawa_rush_01/
```
※ config の `personas:` セクションが空なので、`config_jr_disruption.yaml` の personas をマージする必要あり。または config側にpersonas追記。

### 3-2. shinagawa_integration_notes.md の確認 (保留中)
ユーザー指示で referencedされた `@shinagawa_integration_notes.md §4` が実在しない。現状は config の scene_3d フィールド定義から推測して実装済み。正式な integration_notes が出てきたら rails/roads/decks の描画仕様を照合する。

### 3-3. scene_3d 描画の改善余地 (未着手、優先低)
- **stairs**: 現状は単一の傾斜BoxGeometry。段々表現を追加できる
- **decks のネットワーク接続**: 現状は segments ごとに独立スラブ。`connects_places` 属性で実ビルとの視覚接続を強化可能
- **landmarks の sprite サイズ**: `scale.set(16, 4, 1)` 固定。ビューポート距離でスケール調整するとモバイル視認性改善

### 3-4. モバイル対応 (未着手、低)
grid layout `1fr 340px` なのでスマホでキャンバス狭い。メディアクエリで縦レイアウト切替が要るなら対応。

---

## 4. 復帰用クイックリファレンス

### ファイル位置
- Project root: `D:\ユーザー\デスクトップ\Singulab_huckathon\hackathon\2d-multi-places-simulation-on-fire-public\`
- shinagawa_config.yaml: `D:\ユーザー\ダウンロード\shinagawa_config.yaml`
- viewer_3d_spec.md: `D:\ユーザー\ダウンロード\viewer_3d_spec.md`

### バンドル実行
```bash
# 2D
python tools/bundle_viewer.py simulations/<run_dir>

# 3D (scene_3dなし)
python tools/bundle_viewer.py simulations/<run_dir> --viewer 3d

# 3D (scene_3dあり, shinagawa)
python tools/bundle_viewer.py simulations/<run_dir> --viewer 3d \\
    --scene-3d 'D:\\ユーザー\\ダウンロード\\shinagawa_config.yaml'
```

### git状態
- branch: `feat/stage-1.5-urban-scale`
- Viewer 3D + shinagawa対応は**未commit** (HANDOFF 3-1 実行後にまとめてcommit想定)

### 検証可能なもの
run 13 の `viewer_3d_bundled.html` をブラウザで開けば:
- JR品川駅中心の3D都市描画
- 40エージェントの動きと会話線
- JR運休イベントの赤パルスリング
- shinagawa scene_3d の線路/道路/デッキ/landmarks オーバーレイ
- タップ/クリックでエージェント選択 + 追従カメラ

---

_End of handoff._
