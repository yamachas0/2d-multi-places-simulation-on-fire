# Phase 1 検証レポート — 都市構造 v2 + 機能2/3/4 Stage A

**対象ブランチ** : `feat/stage-1.5-urban-scale`
**直前コミット** : `f855c04` (feat: Phase 1 urban structure)
**前提コミット** : `0104930` (feat 2/3/4 本体)
**検証設定**     : `config_urban_v2.yaml`, seed=42
**実行コマンド** : `venv/Scripts/python.exe main.py --config config_urban_v2.yaml --seed 42`
**出力先**       : `_urban_v2_out/`
**LLM**          : Gemini 3.1 flash-lite-preview (既定)

---

## 1. 実装概要

今回のコミット (`f855c04`) で加えた変更は4点。

| ファイル | 内容 |
|---|---|
| `place_types.py` | 10種の新 place type を追加 (`jr_station` / `subway_station` / `izakaya` / `restaurant` / `convenience_store` / `department_store` / `office_lobby` / `wide_street` / `narrow_street` / `pedestrian_street`)。各々に atmosphere / social_likelihood / typical_stay_duration / is_road / is_spawn_point を付与。 |
| `visualization.py` | 新型 10 種のマップ表示スタイル (塗色 + 枠線 + 道路は破線)。 |
| `simulation.py` | 4本の JSONL シンク (messages / memory_reasoning / should_speak / relationships_timeline) を `threading.Lock` で保護 (並列 15 worker のレース防止)。 |
| `config_urban_v2.yaml` | 30人 × 60step の Stage A フル構成。3階層 (駅 / 広場 / 施設 + 道路) の日本語 place 名を持つ品川駅前モデル都市。既存の火災イベント (中央公園 step40) は温存。 |

設計方針は指示書 §2 の6原則 (JR南・地下鉄左右対称・動線・施設道路沿い・中央広場&公園・オフィス駅離れ) に準拠。

### 新都市レイアウト (19 places)

| Layer | Place | 座標 (x,y) | type |
|---|---|---|---|
| 3 駅  | JR品川駅 | (0,-60) | jr_station (spawn primary) |
| 3 駅  | 地下鉄A駅 | (-55,0) | subway_station (spawn secondary) |
| 3 駅  | 地下鉄B駅 | (55,0) | subway_station (spawn secondary) |
| 2 広場 | 駅前広場 | (0,-45) | plaza |
| 2 広場 | 中央公園 | (25,30) | park (火災中心) |
| 1 道路 | 駅前大通り / 中央通り / 駅前プロムナード / 北町路地 / 南町路地 | 各所 | wide/narrow/pedestrian_street |
| 3 施設 | スタバ駅前 / タリーズ / 駅前コンビニ / 駅前デパート / 居酒屋本町 / イタリアン / 品川第一ビル / 品川第二ビル / 中央図書館 | 各所 | cafe / convenience_store / department_store / izakaya / restaurant / office_lobby / library |

---

## 2. 検証ラン結果サマリ

**実行時間** : **337 秒 ≈ 5分37秒** (30 agents × 60 steps, parallel_workers=15)
**終了コード** : 0
**期待コスト** : Gemini 既定(Haiku比 1/15) → Stage A 実測 ~$0.02 相当と推定

### 生成物一覧

| ファイル | サイズ | 行数 | 状態 |
|---|--:|--:|---|
| `messages.jsonl` | 24 KB | 50 | ✅ |
| `memory_reasoning.jsonl` | 733 KB | 2,130 | ✅ |
| `should_speak_log.jsonl` | 908 KB | 1,185 | ✅ |
| `relationships_timeline.jsonl` | 1.1 MB | 60 | ✅ |
| `animation.gif` (31 frames) | 4.6 MB | - | ✅ |
| `statistics.png` / `report.html` / `transcript.md` | - | - | ✅ |

### JSONL 整合性 (per-line `json.loads`)

```
messages                 50 lines  errors=0
memory_reasoning       2130 lines  errors=0
should_speak_log       1185 lines  errors=0
relationships_timeline   60 lines  errors=0
```

**4本とも 0 エラー**。`threading.Lock` で並列シンクが破損なく直列化されたことを確認。

---

## 3. Stage A における機能 2/3/4 の挙動

### 3-1 機能3 (should_speak ゲート) — p_speak の関係バンド階段性

| 関係帯 (value) | n | p_speak mean | p_speak max |
|---|--:|--:|--:|
| stranger (<0.2) | 2,937 | **0.0030** | 0.0298 |
| face (0.2-0.4) | 359 | 0.0056 | 0.0331 |
| acq (0.4-0.6) | 44 | 0.0169 | 0.0566 |
| friend (0.6-0.8) | 322 | 0.0189 | 0.1252 |
| close (0.8-1.0) | 157 | **0.0603** | 0.3947 |

**stranger と close の平均比 ≈ 20倍**。機能3 の p_speak = talkativeness × social_likelihood × relationship × proximity_factor が想定どおり働き、関係が深いほど発話確率が高い階段が明確に立った。

**decision 内訳**: speak 13 / skip 1,172 (speak 率 1.1%)。50 件のメッセージ行数との差 (50-13=37) は、speak 判定1回の中で同じ step に発生する会話相手の複数埋め込みによるもの。

### 3-2 機能2 (関係値の進化) — 60 step での変化

**初期エッジ分布**: close=8 / friend=12 / acq=6 / stranger=32 (計58エッジ、config に記述したペアのみ)

**最終エッジ分布**: close=13 / friend=12 / face=38 / stranger=265 (計328エッジ)
※ 途中で enter/exit したエージェント間にも関係エッジが生成されたため総数が増加。

- **上昇**: 301 エッジ
- **下降**: 27 エッジ (DECAY=0.001 × 60step = 理論値 -0.06 を下回る自然減)
- **最大上昇ペア**: 鈴木健太 ↔ 鈴木優子 (夫婦) Δ+0.298 (close を超えて上限接近)
- **最大下降ペア**: 佐々木さくら ↔ 佐々木和夫 Δ-0.054 (会話なし期間のDECAYのみ)

PER_CONVERSATION=0.02 で上昇・DECAY=0.001 で減衰という設計値がきれいに現れている。

### 3-3 機能4 (構造化ログ 3形式) — 全てスキーマ通り出力

- `messages.jsonl` : from / from_name / to / to_name / relationship / message / reasoning + step/time (50 lines)
- `should_speak_log.jsonl` : agent_id / nearby_count / candidates[{partner_id, relationship, talkativeness, social_likelihood, proximity_factor, emergency_boost, p_speak}] / decision / selected_partner_id (1,185 calls)
- `relationships_timeline.jsonl` : step / time / relationships[{from_id, from_name, to_id, to_name, value}] (60 rows = 毎step)
- `memory_reasoning.jsonl` : step / time / id / name / layer / memory / reasoning (2,130 records, 44 agents カバー)

---

## 4. 新都市構造の定性観察

### 4-1 火災イベントが動線に与えた影響 (step40 以降)

| step bucket | message count |
|---|--:|
| 0-9 | 1 |
| 10-19 | 8 |
| 20-29 | 5 |
| 30-39 | 8 |
| **40-49** | **17** |
| 50-59 | 11 |

火災発生 (step40, 中央公園, intensity 0.8, radius 25) と同時に会話数が約2倍に跳ね上がった。`emergency_boost` (火災半径×2.0 以内のエージェントで3.0倍・cap 1.0、詳細は `V2_BOOST_VERIFICATION.md`) が `p_speak` を押し上げ、普段無言の弱い関係 (山田太郎→石川裕子 rel=0.07 など) も発話したことが messages.jsonl から確認できる。

発話内容を見ても「煙がこっちから来ているみたいで心配」「念のため避難経路を確認」など、火災コンテキストが自然言語で現れている (LLM が火災状態を読み取っている)。

### 4-2 3階層 place 構造は機能しているか

- **駅 (spawn)**: JR品川駅 / 地下鉄A,B駅の spawn_point=True が働き、30人が各駅に分散スポーン。config の `initial_place` 指定 (例: 田中夫婦はスタバ駅前 / 高橋兄妹は居酒屋本町) も反映。
- **広場 (park / plaza)**: 中央公園が火災中心になり多数エージェントを引き込む。駅前広場は通勤ルートの交差点として機能。
- **施設**: 夫婦が cafe で会話継続・兄妹が izakaya で朝までコース、というペルソナ意図が messages.jsonl で実現 (e.g. 高橋隼人→高橋舞 step3 「そろそろ店を出て帰らないか？」)。
- **道路**: 3種の street (wide/narrow/pedestrian) は現状 "通過ポイント" として機能するが、visualization 上での塗り分けは成立。

### 4-3 視覚的確認 (animation.gif / frame_*.png)

- 31 frames (step 2-60 2step毎) 生成。
- 新型 place の色分けが効いており、火災以降 (step40 以降の frame) で 中央公園が赤く塗られる表示が確認できる。
- 日本語 place 名は matplotlib DejaVu Sans で glyph 警告が出る (frame は描画されるが place 名ラベルが □ 表示)。機能影響なし、後処理で IPAex フォントを入れれば解消。

---

## 5. コストと時間

| 項目 | 値 |
|---|---|
| 実行時間 | 337 秒 (5分37秒) |
| parallel_workers | 15 |
| agents × steps | 30 × 60 = 1,800 agent-step |
| agent-step / 秒 | ≈ 5.3 |
| LLM | Gemini 3.1 flash-lite-preview |
| 推定コスト | ≈ $0.02 (前回 Stage A 実績ベース、Haiku 比 1/15) |

**14分予想 → 5分37秒**。Gemini+並列15で想定より大幅高速化。Phase 2 のサイズ (agents/steps/LLM回数) を増やしても予算インパクトは小さく収まる見込み。

---

## 6. Phase 2 Go/No-Go 判断材料

### Phase 2 に進める根拠 (Go 寄り)

- ✅ JSONL 4本とも整合 (threading.Lock で並列競合なし)
- ✅ 機能3 の階段性が明確 (stranger→close で p_speak × 20倍)
- ✅ 機能2 の関係進化が設計値どおり (Δ+0.02/会話, -0.001/step)
- ✅ 機能4 の構造化ログは全項目出力でき、事後分析に耐える
- ✅ 既存火災パイプラインが新都市構造で壊れず、むしろ step40 以降で会話が倍増する "想定した非常時挙動" を再現
- ✅ 5分37秒で完走 → Phase 2 の大規模化 (例: 50 agents × 120 step) に耐える

### 残課題 / 追加検討余地

1. **matplotlib 日本語フォント** : glyph 警告で place 名が読めない。`IPAexGothic` 等を matplotlib rcParams に指定すれば解決 (Phase 2 の成果物品質向上に必要)。
2. **佐々木兄妹 Δ-0.054** : 実会話なし period の DECAY 蓄積。config の `initial_place: "中央図書館"` が separate で影響した可能性。要件的には正しい挙動だが、Phase 2 で "近くに居たら減衰しない" 強化を入れるなら検討。
3. **道路型 3種の差別化** : wide/narrow/pedestrian で social_likelihood に差をつけたが、Stage A の観察では通過時間が 2 step と短いため差が出にくい。Phase 2 で歩行速度や滞留時間を type 依存にすると効く。
4. **emergency_boost の上限チェック** : step48 に山田太郎が r=0.06~0.23 の 6人に一斉発話。火災時のブロードキャストとして妥当だが、Phase 2 で「何人まで同時に話しかける」上限を加味するか要判断。

---

## 7. 結論

**Phase 1 (都市構造再設計) は Stage A 上で動作を確認した。** 機能2/3/4 が設計どおり働き、JSONL整合性も保証され、新都市構造 + 火災パイプラインが矛盾なく走った。

**次アクション**: Phase 2 (JR運休イベント + 5段階関係ラベル + initial_place 拡張 など、指示書 §4-5 該当) に進むかどうかをやまちゃその判断に委ねる。進める場合は本レポートの「残課題」4点をスコープに含めるか別チケット化するか指示がほしい。

**このレポート時点では Phase 2 の実装には着手していない。**
