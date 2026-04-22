# Singulabo 2D 都市マルチエージェントシミュレーション — 概要（3D化検討用）

他の AI に「この 2D シミュレーションを 3D フィールドに拡張できるか」を判断してもらうための資料。

---

## 1. プロジェクトのひとこと紹介

LLM で駆動するマルチエージェント（最大 40 人規模）の 2D 都市シミュレーション。東京の駅前1ブロック相当のエリアを舞台に、各エージェントがペルソナ・ゴール・関係性を持って歩き・会話し・イベント（JR 運休など）に反応する。Singulabo ハッカソン用のプロトタイプ。

- 本番 seed: 60 step × 30 人で ~25 分実行、Gemini 3.1 flash-lite-preview 使用
- 1 step = 実時間 1 分、1 セル ≈ 5 m、徒歩速度 8 セル/step ≈ 2.4 km/h

---

## 2. 技術スタック

- **言語**: Python 3.13（エージェント/シミュ側）+ HTML5 Canvas + 素の JS（ビューア）
- **LLM**: Gemini / Claude / OpenAI をプロバイダ抽象で切替可能。実運用は Gemini
- **可視化**:
  - `visualization.py` — matplotlib で静止フレーム PNG + GIF + 統計プロット
  - `visualization/viewer_v2.html` — HTML5 Canvas 2D のタイムラインビューア（再生・シーク・速度切替・会話ログ・思考ログ・エージェント追跡）

ファイル規模（抜粋）:
```
agent.py         1734 lines   エージェント・プロンプト生成・行動決定
simulation.py    1631 lines   シム本体・ワールド更新・イベント・ログ・ビューア向け export
visualization.py  695 lines   matplotlib 2D 可視化
main.py           351 lines   CLI エントリ
viewer_v2.html   1238 lines   Canvas 2D ビューア（単一 HTML で完結）
config_jr_disruption.yaml  826 lines   30 persona + 19 places + 時間帯パラメータ
```

---

## 3. ワールドのデータモデル（★ここが 2D ベタ依存）

### 座標系

- 格子セル単位の 2D グリッド。サイズは `half_space_size`（config 値、例: 75）。
- 座標範囲: `X, Y ∈ [-half_space_size, +half_space_size]`
- 全て **整数セル座標**。連続座標ではない。
- エージェント位置は `(x, y)` のみ。**z は存在しない**。

```python
# agent.py: class Agent
self.x, self.y = initial_position  # int, int
```

### 場所（Place）

19 個の axis-aligned な矩形領域。config で矩形を直接書く。

```yaml
- name: "JR品川駅"
  type: "jr_station"
  center_x: 0
  center_y: -60
  half_size_x: 18   # 矩形の半幅（X方向）
  half_size_y: 5    # 矩形の半幅（Y方向）
  capacity: 40
  attributes: {...}
```

type は以下の 14 種（どれも 2D 意味論）:

```
jr_station, subway_station,
plaza, park,
cafe, restaurant, izakaya,
convenience_store, department_store,
office_lobby, library,
narrow_street, wide_street, pedestrian_street
```

レイヤ概念はあるが、これは **レンダリング描画順序のみ**（1=道路 下, 2=広場/公園, 3=施設 上）。物理的な高さ（階）ではない。

### 時間

- `start_time: "17:00"` から 1 step = 1 分で進行
- 時間帯パターンで `enter_per_step`（駅からの新規スポーン率）・`exit_per_step` を切替

---

## 4. エージェントの行動（2D 前提）

### 移動アクション

LLM に与える移動の意味論は 2D 限定:
- `walk_toward(target_x, target_y)` — 指定座標方向へ base_cells_per_step セル進む
- `walk_along(direction)` — direction ∈ {up, down, left, right, ...}
- `approach(agent_id)` / `approach(place_name)` — 対象方向へ
- `enter(place_name)` — 場所内に入る
- `stay` — その場でとどまる

プロンプトのフィールド境界説明は agent.py:635 / 961 / 1172 で 3 回出現、全て 2D 前提:

```
Field boundaries: X and Y both range from -{half_space_size} to +{half_space_size} inclusive.
Scale: 1 cell ≈ 5 meters, so the full map spans roughly {half_space_size * 2 * 5} meters on each side.
```

境界クランプも 2D:

```python
# agent.py
max(-half_space_size, min(half_space_size, x)),
max(-half_space_size, min(half_space_size, y)),
```

### 行動レイヤ（LLM 呼び出しコスト最適化用）

- `transit` — 場所の外を歩いている。キャッシュで低コスト
- `dwelling` — 場所に入って留まっている
- `interacting` — 誰かと会話中

これは 3D とは独立の概念。

---

## 5. 認識・会話・イベント（3D 化の影響は小）

- `communication_radius: 10` セル（直線距離でフィルタ）
- 会話は 1:1 の partner 制（先日修正済の「方針D」）
- イベント伝搬 3 経路: `notification`（通知）/ `direct`（物理的に近い）/ `conversation`（会話で伝聞）
- JR 運休イベント・火災イベント（disruption type）を持つ

これらは座標空間に依存するが、距離計算と「場所に入っているか」判定だけなので、3D 化しても `math.hypot(dx, dy, dz)` や矩形→直方体の点内判定に置き換えるだけで動く。

---

## 6. エクスポートされるデータ（ビューア入力）

`sim.export_simulation_data()` が `simulation_data.json` を書き出す。構造:

```json
{
  "metadata": {
    "half_space_size": 75,
    "meters_per_cell": 5
  },
  "personas": [ { "id": 0, "name": "田中健一", "gender": "male", ... } ],
  "places": [
    { "name": "JR品川駅", "type": "jr_station",
      "center_x": 0, "center_y": -60,
      "half_size_x": 18, "half_size_y": 5 }
  ],
  "timeline": [
    {
      "step": 0, "time": "17:00",
      "agents": [ { "id": 0, "x": 10, "y": -55,
                    "layer": "dwelling", "in_place": true,
                    "awareness_source": {"jr_shinagawa_outage": "direct"} } ],
      "conversations": [ { "from": 0, "to": 1, "message": "..." } ],
      "events": [ { "type": "transit_disruption",
                    "affected_place": "JR品川駅",
                    "remaining_steps": 4 } ]
    },
    ...
  ]
}
```

**ここにも z はない。** 3D 化するなら
- `metadata.height_range` 的な値の追加
- `places[i]` に `center_z`, `half_size_z` 追加（あるいは 2D のまま「階」属性）
- `timeline[i].agents[j].z` 追加

の変更が必要。

---

## 7. 現在の可視化（2D）

### matplotlib（静止フレーム + GIF）

- `FIGURE_SIZE = (12, 13)`（正方形マップ + 場所別占有率の下段フッター）
- 2D 平面に矩形塗り + 円マーカー（エージェント）+ 破線（会話）+ 運休バッジ

### HTML5 Canvas ビューア (viewer_v2.html)

- `ctx.arc()` でエージェント円を描画、`fillRect` で場所矩形
- ステップ間補間: `state.stepFrac` と `lerpAgentPos()` でフレーム間を滑らかに
- 右ペイン: 会話ログ / 思考ログ / フォーカス詳細
- 時間帯ごとの背景グラデーション（夕方=濃い pine 緑、夜=ほぼ黒）

全て 2D 描画 API。**Three.js / WebGL などの 3D パイプラインは未導入**。

---

## 8. 3D 化を考えるときの主な論点

### A. 座標・矩形の拡張（機械的）
- 全所 `(x, y)` → `(x, y, z)` への拡張
- Place を直方体 (`center_z`, `half_size_z`) にするか、「階」属性にするか
- 境界クランプと距離計算の 3D 化

### B. LLM プロンプトの書き換え（中程度）
- "Field boundaries: X and Y both range..." を 3D に
- 移動アクション（up/down/left/right）に forward/back/up/down などの z 軸系を足すか、あるいは "go to floor 3" 的なハイレベル指示にするか
- 3 箇所（agent.py:635 / 961 / 1172）の境界説明文を更新
- プロンプトが長くなり LLM コストに影響する

### C. 可視化の総取替（大）
- matplotlib を Axes3D にするか、静止は諦める判断
- Canvas ビューアは Three.js / Babylon.js / Deck.gl へ全書き換え
- 現 viewer_v2.html の 3 ペイン構成・思考ログ・追跡 UI 等は再利用可能だが、描画部分だけ差し替え
- スポーン/despawn アニメ、会話線、イベントハイライト、フォーカストレイルを 3D 化する工数

### D. シナリオの意味論（要判断）
- 現シナリオは駅前1ブロック相当で「平面都市」前提。**3D にして面白くなるか** は要検討
- 例: オフィスビルの階層移動、地下鉄の地上/地下、避難経路の高さ差
- 「階」属性 + 2D マップだけで済むなら、完全 3D 化しなくても多くのユースケースを表現できる

### E. データサイズ・パフォーマンス
- 30 人 × 60 step でビューアバンドル 1.16MB（JSON インライン）
- 3D 化で座標が増えても微増
- WebGL レンダは 30 人程度なら余裕

---

## 9. 「最小 3D 化」提案（もし段階的にやるなら）

1. **疑似 3D**: 現 2D ロジックは据え置き、Canvas ビューアだけ Three.js の等角カメラ or 俯瞰カメラに差し替え。Place を高さ付き箱で描画し、エージェントは床面を這う。コスト: 可視化の書き換えのみ
2. **階のみ追加**: エージェント・Place に integer `floor` 属性を足す。LLM プロンプトに「いま何階」を含めるだけで、物理的な 3D 空間は保持しない。コスト: 中
3. **フル 3D**: 上記 A〜C 全部。コスト: 大

---

## 10. 参考: リポジトリ構成

```
hackathon/2d-multi-places-simulation-on-fire-public/
├── agent.py                     LLM プロンプト・行動決定・エージェント状態
├── simulation.py                ワールドループ・イベント・ビューア export
├── visualization.py             matplotlib ビジュアライザ（2D）
├── main.py                      CLI エントリ
├── config_jr_disruption.yaml    Phase 3 本番 config（30 人 × 60 step）
├── claude_client.py / gemini_client.py / openai_client.py
├── visualization/
│   └── viewer_v2.html           Canvas 2D タイムラインビューア
├── tools/
│   ├── bundle_viewer.py         simulation_data.json を viewer にインライン
│   ├── analyze_duplicate_messages.py
│   └── scan_time_vocab.py
└── simulations/                 ランごとのログ（gitignored）
    └── YYYY-MM-DD_HHMM_NN_name/
        ├── simulation_data.json
        ├── messages.jsonl
        ├── awareness_propagation_log.jsonl
        ├── should_speak_log.jsonl
        ├── memory_reasoning.jsonl
        └── viewer_v2_bundled.html
```

---

## 結論（他 AI への投げかけ）

「この 2D シミュレーションを 3D フィールドに拡張できるか？」に対する回答として期待したいこと:

- 3D 化の最小実装案（座標だけ 3D化 / 可視化だけ 3D化 / 完全 3D化 のどのレベルが妥当か）
- LLM プロンプトで 3D 空間をどう記述すべきか（up/down 指示、高さの認知、階層のメンタルモデル）
- Three.js / Babylon.js / Deck.gl / react-three-fiber の中で Canvas viewer の差し替え先として最適なのはどれか
- シナリオ上、3D 化で本当に価値が出るユースケースの例
- 工数見積（ハッカソン残時間で可能な範囲）
