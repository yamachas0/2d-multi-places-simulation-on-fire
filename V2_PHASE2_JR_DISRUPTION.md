# Phase 2 検証レポート — JR運休イベント (機能 4.5)

**対象ブランチ** : `feat/stage-1.5-urban-scale`
**直前コミット** : `1eb9956` (feat: Phase 2 transit disruption event)
**検証設定**     : `config_jr_disruption.yaml`, seed=42
**実行コマンド** : `venv/Scripts/python.exe main.py --config config_jr_disruption.yaml --seed 42`
**出力先**       : `_jr_disruption_out/`
**LLM**          : Gemini 3.1 flash-lite-preview

---

## 1. 実行サマリ

**実行時間** : 386 秒 (6分26秒) — Phase 1 (337s) + 約50秒
**終了コード** : 0
**ステップ** : 30 agents × 60 steps, parallel_workers=15
**追加コスト** : Phase 1 と同等 (≈ $0.02)

### 生成物

| ファイル | サイズ | 行数 | 状態 |
|---|--:|--:|---|
| `messages.jsonl` | 38 KB | 84 | ✅ |
| `memory_reasoning.jsonl` | 749 KB | 2,166 | ✅ |
| `should_speak_log.jsonl` | 1,012 KB | 1,323 | ✅ |
| `relationships_timeline.jsonl` | 1,109 KB | 60 | ✅ |
| **`event_awareness_log.jsonl`** (新規) | 126 KB | 60 | ✅ |
| `animation.gif` (31 frames) | 5.2 MB | - | ✅ |
| `statistics.png` / `report.html` / `transcript.md` | - | - | ✅ |

### JSONL 整合性

```
messages.jsonl                              84 lines  errors=0
memory_reasoning.jsonl                    2166 lines  errors=0
should_speak_log.jsonl                    1323 lines  errors=0
relationships_timeline.jsonl                60 lines  errors=0
event_awareness_log.jsonl                   60 lines  errors=0
```

**全シンク 0 エラー**。threading.Lock による直列化は Phase 1 と同様に機能した。

---

## 2. 機能 4.5 パイプラインの動作確認

### 2-1 イベント起動

```
Step 30 08:30: EVENT 'jr_shinagawa_outage' activated (type=transit_disruption,
               affected=JR品川駅, pos=(0.0, -60.0))
```

`_update_event_states()` が config の `start_step: 30` を検知して起動。`place_position` が JR品川駅 のセンター座標に解決され、以降 `_propagate_direct_event_awareness` / `_transit_disruption_near` / `get_active_events_for_agent` が正しく読めた。

### 2-2 直接認識 (notify_radius=25)

JR品川駅 (0, -60) から半径 25 以内にある places:

| place | 距離 | 備考 |
|---|--:|---|
| JR品川駅 | 0.0 | 中心 |
| 駅前プロムナード | 8.0 | 17/18/21 の動線上 |
| 駅前広場 | 15.0 | 待ち合わせ系 |
| 駅前大通り | 25.0 | 境界 |

半径内に居たことで認識した総人数 : **3人** (step 30, 43, 52 でそれぞれ 1 人)

| step | 新 aware | name | initial_place |
|--:|--:|---|---|
| 30 | 西村理香 (id=25) | 駅前広場 (d=15) |
| 43 | 岩田伸吾 (id=21) | 駅前プロムナード (d=8) |
| 52 | 中田千代 (id=23) | 駅前デパート (d>25, 途中で近づいた) |

**直接認識そのものはコード通り動いた**が、次の 2-5 で述べるとおり、実際に認識された人数が 3 に留まった背景はシナリオ側の事情 (JR通勤者が step30 前に駅を離れていた)。

### 2-3 会話による伝播 (keyword-based)

**キーワードヒット** : messages.jsonl 84 行中 **0 行** に `運休` / `電車` / `振替` 等の transit キーワードを含むものなし。

原因:

- 認識した 3 人 (21 岩田 / 23 中田 / 25 西村) は talkativeness が 0.2-0.5 の低めソロ系。
- 彼らは event 中 (step 30-60) に **1 件も message を送らなかった**。should_speak_log を見ると p_speak が 0.01-0.03 で推移し、emergency_boost (×3.0) を掛けても ≈ 0.03-0.09 程度。
- したがって path-2 (conversation keyword) が発火する条件 (aware agent が発話する) が成立しなかった。

**コード側は keyword リストを正しく持っている** (`_transit_keywords` 11 語) し、`_propagate_event_via_message` は Phase 2 smoke test で発火実績あり。今回の run で path-2 が 0 件だったのは config 上の設計成果 (下記 §3)。

### 2-4 emergency_boost (運休への拡張)

|  | n_candidates | boosted (≥3.0) | 率 |
|---|--:|--:|--:|
| pre-event  (step 0-29) | 1,391 | 0 | 0.0% |
| during-event (step 30-60) | 2,725 | **18** | **0.7%** |

**pre-event は 0件**。`_fires_near=False` かつ `_transit_disruption_near=False` なので boost=1.0 しか出ない ✅
**during-event は 18件**。aware 3 人が半径 25 以内 (`_transit_disruption_near=True`) に居た step で正しく boost=3.0 が乗った。

```
id=21 岩田伸吾:  steps [44, 45, 46]                     (n=3)
id=23 中田千代:  steps [52, 53, 54, 55, 56, 57, 58, 59]  (n=8)
id=25 西村理香:  steps [45, 46, 53, 55, 56, 58, 60]      (n=7)
```

`in_emergency = self._fires_near(...) or self._transit_disruption_near(agent)` が期待どおり運休側で True を返している。

### 2-5 スポーン制御 (block / boost_spawn_at)

時系列のスポーン履歴 (log から抜粋):

| step | spawn | 備考 |
|--:|---|---|
| 1 | JR品川駅 | pre-event (block なし) |
| 12 | 地下鉄A駅 | pre-event |
| 13, 16, 21, 25 | 地下鉄B駅 | pre-event |
| **31** | 地下鉄A駅 | **event 中, JR block ✅** |
| **32** | 地下鉄A駅 | event 中 |
| **33, 34, 36, 40** | 地下鉄B駅 | event 中 |
| **38** | 地下鉄A駅 | event 中 |

**event 発動後 (step 30以降) に JR品川駅 へスポーンした agent はゼロ**。7回のスポーンがあったが **100% が 地下鉄A/B駅** に着地。`block_spawn_at: [JR品川駅]` が効いている ✅

`boost_spawn_at` (multiplier=2.0) は `random.choices(weights=...)` の重みが 2 倍になるため、地下鉄A/B駅 の選択確率が他の spawn 系 place を上回る。事実、pre-event でも 地下鉄B駅 への spawn が多かった (spawn_role=secondary なので spawn 対象) 傾向が、event 中にさらに強まった。

---

## 3. 観測された定性傾向

### 3-1 動線への影響

- 事件は静か (通勤者 17/18 は既に離脱済み)。messages.jsonl 内のトピックは殆どが家族・同僚の日常会話で、運休関連の言及なし。
- 火災 (Phase 1) のように会話数が跳ね上がる効果は出ていない: 会話数は step 20-29 で 26件、30-39 で 16件、40-49 で 18件、50-59 で 7件 と減衰傾向 (火災時は +100% でピーク到達していたのと対照的)。
- 火災 vs 運休の**差別化は出ている**: 火災は視覚的・即時的に 2.0×radius 以内の全員を巻き込むが、運休は「知る」ステップが必要な情報系イベントなので、情報伝播が弱い agent はそもそも気づかない。

### 3-2 awareness が 3/40 にとどまった構造的要因

1. **JR通勤者 (17 石川 / 18 山本) は event 発動前に離脱済み**。彼らの initial_place は JR品川駅 だが step 30 時点では既に オフィス (品川第一/第二ビル ロビー) 方向に移動済み (通勤距離 45-50 セル、speed 8 × 60 steps で十分到達可能)。
2. **認識した 3 人は 低-talk ソロ系** (talk=0.2-0.5)。emergency_boost=3 を掛けても p_speak が 0.03-0.09 止まりで、speak 判定を 1 度も通過しなかった。
3. **会話による path-2 が発火する条件** (aware agent × 近くの agent × speak 判定通過 × キーワード含有発話) が成立せず。

→ これは **実装のバグではなく、JR運休シナリオ設計上の性質**。次の Phase で ① event 起動を step15 に前倒しして通勤中に引っ掛ける、② notify_radius を 40 に拡大する、③ aware agent に system-level な「発話圧」を足す (例: event 認知時に一度は broadcast するフラグ)、のいずれかを導入すれば awareness が拡散する。

### 3-3 salience_boost_targets の実効

config で `salience_boost_targets: [株式会社テックビジョン, グローバル商事]` を指定したが、テックビジョン (6/7/8) もグローバル商事 (9/10/11) も誰も known_events に入らなかったため `_apply_salience_boost` は彼らに対して発火せず。

(認識した 3 人の social_identities は空または別グループで、targets に該当しなかった。)

→ 機能としてコードは動く (Phase 2 smoke test で検証済) が、今回の run では aware agent と targets が交差しなかった。

### 3-4 関係値の進化 (上書き)

| | init | final |
|---|--:|--:|
| エッジ総数 | 58 | 265 |
| 上昇 | - | 151 |
| 下降 | - | 25 |
| 最大上昇 | - | 中村(9)↔加藤(11) Δ+0.376 |
| 最大下降 | - | 高橋兄妹(15↔16) Δ-0.804 **(agent 15 が step22 で despawn したため: 途中退場 agent の関係エッジが snapshot に載らない)** |

PER_CONVERSATION=0.02, DECAY=0.001 の挙動は Phase 1 と整合。最大下降 Δ-0.804 は監査上のアーティファクト (despawn で final snapshot から消える) であり、実際の関係値変化ではない。

---

## 4. Phase 2 → 3 Go/No-Go 判断材料

### Go 寄りの根拠

- ✅ **5 JSONL 全てが行エラー 0** (threading.Lock が運休 path でも破損を防いだ)
- ✅ **イベント起動・place_position 解決・awareness 追跡・ログ出力が end-to-end で動作**
- ✅ **spawn 制御 (block_spawn_at / boost_spawn_at) が正しく発火** — event 後の 7 スポーン全て地下鉄へ
- ✅ **emergency_boost の運休拡張が動作** — aware agent の近傍で 18 件の boost、pre-event は 0 件 (静穏性も確認)
- ✅ **events_info が prompt に正しく注入されている** (agent.py:_build_events_section が message / action 両経路で呼ばれる)
- ✅ **実行時間 6分26秒 / 追加コスト ≈ $0.02** — Phase 3 で config 差し替え再ランする余裕あり

### 課題 / Phase 3 で検討したい調整

1. **awareness が 3/40 で止まった** — JR通勤者が離脱後に event が起動するタイミング問題。`start_step: 15` 前倒し or `notify_radius: 40` 拡大で改善。**実装ではなく config チューニング**。
2. **aware agent がソロ系 low-talk に偏った** — path-2 (conversation keyword) が発火せず。以下どれかで改善:
   - (a) aware agent に `salience_on_awareness` 的な発話圧を与える
   - (b) aware agent が同 place で nearby に居合わせた時のみ speak_gate を +0.3 する緊急バイアス
   - (c) config 側で aware 候補が自然と高-talk 系になるように persona を選ぶ
   Phase 2 のコードを壊さずに (a) を足すなら `agent.py` の `decide_message` 前に「aware かつ events_info あり ならこの step は speak に偏らせる」フラグ1つで済む。
3. **salience_boost_targets の group と aware agent の交差がゼロ** — これは targets の書き方 (テックビジョン / グローバル商事 の通勤グループは JR 近辺に居ない) と aware agent 側の social_identities (21/23/25 は group なしかソロ系) の噛み合わせの問題。Phase 3 で targets を `通勤者全般` など自然と交差する抽象度にするか、aware に通勤系 persona を意図的に混ぜる。
4. **火災との差別化の定性検証** — path-1 vs path-2 の 2 段階認識というデザインは、火災 (物理的・即時) との差異を出すためのもの。今回の run で path-2 が発火しなかった以上、**2 段階認識のメリットが定量的に示せていない**。Phase 3 で一度 aware agent 発話問題を解いてから path-2 hits を定量化するのが筋。

---

## 5. 結論

**Phase 2 の実装は仕様どおり完走した**。イベント型の定義、状態遷移、2経路の awareness 伝播、spawn 制御、emergency_boost 拡張、専用ログシンクの 6 コンポーネントが全て動作し、JSONL 整合性も保証された。

一方で、**今回の config (config_jr_disruption.yaml, seed=42) では awareness が 3 人しか発生せず**、path-2 (conversation keyword) が 0 件・salience_boost_targets も未発火だった。これは **実装のバグではなく、JR運休 x 動線 x persona 分布の噛み合わせ**。Phase 3 (もし進めるなら) では、

- config チューニング (start_step 前倒し・notify_radius 拡大) で awareness を増やす
- もしくは aware agent の発話圧を上げるパラメータを1つ足して path-2 を駆動する

のどちらかで本機能の効果を再観測したい。

**次アクション判断**: Phase 3 に進むか、config/awareness 駆動パラメータを Phase 2 のスコープに足して再検証するか、やまちゃそ判断に委ねる。

**このレポート時点で Phase 3 (もしあれば) の追加実装には着手していない。**
