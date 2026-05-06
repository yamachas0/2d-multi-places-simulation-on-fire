# シミュレーション情報フロー俯瞰 (smoke4 時点)

「どの情報が、いつ、どこに渡され、どう発話/移動のトリガーになっているか」を Phase A → B → C の時系列で整理。

---

## 0. 共通の前提 (全フェーズに効くもの)

### エージェントの基本構造
- **persona dict**: name / age / gender / occupation / background / speech_style / catchphrase / temperament_3軸 / current_goal / talkativeness / phone_check_rate / initial_place / axis_id / variant / school_fit / interest_tag / tendency / cognitive_biases / initial_relationships
- **state**:
  - `position` (x,y) ─ Phase B のみ動く
  - `internal_state` ─ energy / hunger / social_fatigue
  - `relationships` ─ 各 agent ごと 0.0〜1.0 の親密度。会話成立で +Δ
  - `memory` ─ 文字列リスト。最新 `memory_size` 件だけが prompt に乗る (rolling buffer)
  - `sent_messages` / `received_messages` ─ 直近の発話履歴 (rolling buffer)
  - `visited_places` / `entered_places` ─ Phase B で perceive_pass / perceive_enter を注入した place の set

### 発話確率の数式 (Phase A/B 共通)
```
p_speak = talkativeness × social_likelihood × relationship × proximity_factor × 各種ブースト
       (上限 1.0、毎step判定)
```
- `proximity_factor`: dist≤2→1.0 / dist≤4→0.4 / それ以上→0.1
- `proximity_close_boost`: dist≤2 のとき ×1.5
- `companion_boost`: dist≤2 かつ relationship≥0.5 のとき ×1.5 (重ね掛け最大2.25x)
- `fatigue_factor`: max(0.2, 1 - social_fatigue/200) で抑制
- `emergency_boost`: 火災等で ×3
- **host idle 抑制**: is_host=True かつ近接に学生いない → 即 silent (LLM call スキップ)

### 行動確率 (Phase B のみ)
- 行動 phase の LLM call で `walk_toward / walk_along / enter / stay / approach / wander` から **LLM が選ぶ**
- 確率式は無し、LLM の文脈判断のみ
- **navigation.py の walkable mask** 上のみ移動可。1 step = 最大 12 cells (≒60m)
- 道路幅員 ≥ 5cells の道は両端歩道のみ歩行可

---

## 1. Phase A 開始時 (t=0、9:00)

### 各生徒に渡される情報

**system_prompt (cache 領域、毎step同じ)**
```
You are an autonomous agent in <a classroom in Tokyo>.
=== MESSAGE RULES === (broadcast / 200 words / 沈黙OK)
=== CONVERSATION PRINCIPLES === (relationship label による tone)
=== RESPONSE FORMAT === JSON
=== BEHAVIORAL GUIDANCE === (沈黙OK / 重複禁止)
=== ON-SITE PERCEPTION OF THE TOWN BEING DISCUSSED ===
  ↑ ここに `inject_context` で品川GL2020 の事前情報が全文注入される
=== CENTRAL QUESTION (中心問い) ===
  「このまちのいいところ・悪いところを話し合い、こうなったらいいという未来像を描いてください。」
```

**user_prompt (毎step変動)**
- persona_section (名前/年齢/性別/職業/気質/背景/口癖/speech_style/cognitive_biases) ※ current_goal は外した、system_prompt に移動済み
- internal_state (energy/hunger/social_fatigue)
- time_section (Current time / Neighborhood mood / 言語ガード)
- NEARBY PEOPLE (educator が居る or 隣の席など。relationship label と status のみ、座標は省略)
- PREVIOUS MEMORY (memory rolling buffer の最後 N 件)
- RECENT CONVERSATION (受信+送信のマージ window)
- Step

**initial_memory** ─ smoke4 では **空** (smoke3 で CLASS_TASK_NOTE 入れたが副作用で外した)

### 設定 (config_classroom_ab_*.yaml)
- `minimal_prompt_mode: true` → 上記 minimal 版 prompt 使用
- `skip_decision_prompt: true` → Phase 3 (action) を skip。発話 phase で memory 直接書き込み
- `talkativeness: 0.66` (1.2× 引き上げ)
- `communication_radius: 12` (教室全体届く)
- `skip_probability: 0.05` (5% は完全にスキップ)
- `step_duration_minutes: 2`、`start_time: 09:00`、duration 30 step (smoke) / 100 step (本番)

---

## 2. Phase A 各 step (t=1..30/100)

毎step `simulation.step_simulation()` で以下:

```
[step N 開始]
↓
update_state()  ← internal_state を更新 (会話で疲労蓄積、待機で休息)
↓
Phase 1: 発話判定 + 発話生成
  各 agent ごと並列で:
    nearby_agents 取得
    should_speak ゲート (確率判定)  ← LLM call の前段
      → 当たれば LLM call (decide_message)
      → 外れれば silent  (LLM 非実行、コスト 0)
    decide_message:
      system_prompt + user_prompt を組み立てて Gemini 呼び出し
      JSON 返答 → message + reasoning + memory を抽出
↓
Phase 2: メッセージ delivery
  発話成立した agent の message を 1対1 で partner に送る
  (event-based propagate で他人にも噂が届く)
↓
Phase 3: 決定 phase
  Phase A は skip_decision_prompt=True なので、stub action_type=stay 強制。
  memory は Phase 1 で既に書き込み済み。
↓
[step N 終了]
relationships 更新、memory ログ書き込み
```

### Phase A の発話/思考のトリガー
- **発話**: `should_speak` が当たる (確率) かつ relationship・近接条件
- **思考(memory)**: 毎 step LLM call があれば自動で memory + reasoning が書き込まれる (skip_decision で1 call で両方)
- **特定キーワードに反応する仕組みは無い**。LLM が文脈 (中心問い・PREVIOUS MEMORY・RECENT CONVERSATION) から判断
- 移動: なし (教室固定、座席に座ってる)

### 観察された問題
- 全員が **同じ GL2020 事前情報** + 同じ中心問い → 同じ言葉に収束 (「ごちゃ混ぜ感」連呼 13/24)
- ペルソナの多様性 (interest_tag/school_fit/tendency) が**会話の多様性に変換されてない**
- 相互に意見が違う構造になっていない

---

## 3. Phase A → Phase B 引継ぎ

### 抽出ツール (Phase A run_dir に対して実行)

**`tools/extract_fw_handoff.py`** ─ 各生徒の Phase A 全 memory + 全 messages を Gemini に渡して:
- `future_image` (1文): 座学を経て描いた品川の未来像
- `intent` (1文): FW でこう過ごしたい
- `key_memories` (3つ): 座学で気になっていたこと

**`tools/extract_field_questions.py`** ─ 同データから:
- `field_questions` (3問): FW で確かめたいこと
- `one_liner` (1文): FW全体の自分のテーマ

→ `fw_handoff.jsonl` / `field_questions.jsonl` を Phase A run_dir に保存

### 引継ぎ点で **失われる情報**
- Phase A の細かい会話の流れ・ニュアンス
- ペルソナ間の相互作用 (誰と仲良くなったか等)
- relationships は次フェーズで作り直し (axis_id 単位で 0.55 中立リセット)

→ Phase A での発見は **5項目 (future_image/intent/3つの key_memories) + 4項目 (field_questions/one_liner) = 9項目に圧縮**

---

## 4. Phase B 開始時 (t=0、13:00)

### 各生徒に渡される情報

**persona dict** (Phase A から copy + 一部上書き):
- `initial_place` を `港南口広場` (SPAWN_PLACE_NAME) に変更
- `current_goal` を **FW_GOAL** に変更:
  > 「このまちのいいところや課題を様々な視点で探し、こうなったらいいと思う未来像を描いてください。」
- `talkativeness` を +0.10 (FW中は積極的) → 0.66 → 0.76
- `field_questions` (3問) を持ち越し ※ smoke4 では system_prompt 固定挿入は外し、memory ルートのみ
- `initial_memory` (build_initial_memory):

```
[座学を経て描いた品川の未来像] {future_image}
[これから現地でどう過ごしたいか] {intent}
[座学で気になっていること 1] {key_memories[0]}
[座学で気になっていること 2] {key_memories[1]}
[座学で気になっていること 3] {key_memories[2]}
[現地で自分の目で確かめたい 1] {field_questions[0]}
[現地で自分の目で確かめたい 2] {field_questions[1]}
[現地で自分の目で確かめたい 3] {field_questions[2]}
[フィールドワーク全体の自分のテーマ] {one_liner}
[前提] これは企業協力型の学外教育プログラム… (FW_PREMISE 全文)
[今日のFW課題] FW中、最低 2社以上の企業担当者を訪ね、… (FW_TASK 全文)
[現在地] 港南口広場。13時に集合してフィールドワークを始めたばかり。
```

**重要: FW_TASK は initial_memory に**1度だけ**注入される**。memory rolling buffer (memory_size 件) からは時間とともに押し出される → **後半 step では FW_TASK が prompt に乗らない可能性大**

**system_prompt (full 版)**
```
You are an autonomous agent in <Shinagawa>.
=== WORLD STRUCTURE === (座標系/通信半径10/通信ルール)
=== PLACE LOCATIONS === (全 place の座標+capacity)
=== DATA INTERPRETATION === (定量データ解釈)
=== MESSAGE RULES === (broadcast/200words/沈黙OK)
=== CONVERSATION PRINCIPLES === (関係性 tone)
=== WHEN TRANSIT IS DISRUPTED === (電車運休時の判断)
=== RESPONSE FORMAT === JSON
=== EXTENDED GUIDANCE === (10sub-section: 位置秘匿/turn-taking/思考と発話の差/etc)
=== CONVERSATION STYLE GUIDELINES === (DO/DONT)
=== CENTRAL QUESTION (中心問い) ===
  FW_GOAL ← 中心問い
```

### host 12人 にも同様
- `initial_place` = 各社の施設名 (品川インターシティ等)
- `initial_position` = 最寄り道路の歩道座標 (build時計算)
- `is_host: true`
- `current_goal` = 動機型 (smoke4 で書き換え):
  > 「あなたは…単なる施設の説明員ではなく、この街の運営に関わる一人の当事者です。あなた自身、自社と街が、若い人や未来の利用者からどう見られているかを知りたい立場にあります。…」

### 設定 (config_phaseB_*.yaml)
- `minimal_prompt_mode: false` → full prompt
- `skip_decision_prompt: false` → action phase 復活
- `talkativeness: 0.76`
- `communication_radius: 10` (≒50m)
- `step_duration_minutes: 2`、`start_time: 13:00`、duration 50 step (smoke) / 100 step (本番)

---

## 5. Phase B 各 step (t=1..50/100)

```
[step N 開始]
↓
update_state()
↓
_phase_b_inject_perceive(action_decisions of prev step):
  agent の position が place 内に入った場合、その place の perceive_pass / perceive_enter を memory に追加
  (visited_places / entered_places set で「初回のみ」を制御)
↓
_phase_b_log_body_sense(): N step ごとに「累積距離・時間・近隣環境」を memory に追加 (事実のみ、疲労はLLMに任せる)
↓
Phase 1: 発話判定 + 発話生成 (上記と同じ)
  ※ host は近接に学生いないと silent (LLM call スキップ)
↓
Phase 2: delivery + propagate
↓
Phase 3: 決定 phase (Phase A と異なり実行)
  各 agent ごと並列で decide_action (LLM call):
    full prompt builder で:
      system_prompt: WORLD STRUCTURE / PLACE LOCATIONS / AVAILABLE ACTIONS / RESPONSE FORMAT / DETAILED SEMANTICS / EXTENDED GUIDANCE / CENTRAL QUESTION
      user_prompt: persona + internal + time + state(現x,y) + nearby_places + nearby_agents + memory + messages + step
    JSON 返答 → action_type / target_place / target_agent / direction / memory / reasoning
  navigation.py が action_type を移動座標に変換 (walkable mask 上のみ、最大12cells)
↓
[step N 終了]
relationships 更新、memory ログ
```

### memory に毎step 自動挿入される rule-based prefix (smoke3 以降)
```
[訪問履歴] 入場済み: A, B / 通過済み(未入場): C, D
↓
- (memory rolling buffer 最新 N 件)
```
※ rolling buffer の外で固定。同じ場所への無自覚な再訪を抑制。

### Phase B の発話/移動のトリガー
- **発話**: `should_speak` 確率 + nearby (Phase A と同じ + companion_boost)
- **移動**: action phase で LLM が `walk_toward / walk_along / enter / stay / approach / wander` から選ぶ
  - 動機: 中心問い + memory + nearby_places (近くの場所一覧) から LLM が文脈判断
  - **特定の "ここに行け" トリガーは無い** (FW_TASK は memory にあるだけ、後半は消える)
  - field_questions も memory にしか無い
- **思考 (memory)**: 行動 phase で書き込み (発話 phase と二重に書く構造)

### 観察された問題
- **2社以上 host 訪問達成: 0/10** ← FW_TASK が無視された
- 港南口広場 / 自由通路 / 住宅 / カフェ に居座り傾向
- 「初対面の企業に行く動機」が弱い

---

## 6. Phase B → Phase C 引継ぎ

何もしない。Phase C は Phase B run_dir を直接読む。

---

## 7. Phase C アンケート (run_survey.py)

各 agent (生徒10 + host12 = 22) ごとに、独立した Gemini call:

**system_prompt** (生徒/host で別)
- 「あなたはシミュレーションに参加した『生徒』 (or 『企業の受入担当』) です。シミュレーションが終了したいま、運営者から渡されたアンケートに、自分の体験を踏まえて率直に答えてください。」
- JSON 出力フォーマット指定 (10段階整数 + 自由記述)

**user_prompt**
- 「あなた」(name/年齢/役/職業)
- 背景 (persona.background[:500])
- Phase A のあなたの思考記録 (memory_reasoning.jsonl から `[step N] mem: … | reason: …` を最大30件)
- Phase B のあなたの思考記録 (同様、最大30件)
- Phase B でのあなたの発話 (最大20件)
- Phase B であなたが受信した発話 (最大20件)
- アンケート設問 (10段階+自由記述)

→ `survey_responses_student.jsonl` / `survey_responses_host.jsonl` 出力

### 数値設問
- 生徒 Q1 街の印象変化 (1=印象悪化 / 10=大幅向上)
- 生徒 Q2 視野が広がった実感 (1=なし / 10=強くある)
- 生徒 Q3 街に「自分も関われる」と思えた度合い (1=思えない / 10=思える)
- 企業 Q7 学生との対話の手応え (1=響かなかった / 10=深く議論できた)
- 企業 Q8 自社にとっての価値 (1=価値なし / 10=継続したい)

### 観察された問題
- LLM の **「経験をポジティブ翻訳する」忖度バイアス**
- 数値だけ見ると「成功した」ように見えるが、**実際の actual behavior (移動・発話の質) と乖離**
- Phase B の memory に「host と少し話した」事実があると Q7=6 と書く傾向

---

## 8. 全体まとめ (発話/移動の主要トリガー)

| トリガー種別 | Phase A | Phase B |
|---|---|---|
| 発話 | should_speak 確率 | should_speak 確率 |
| ── 確率の主要因 | talkativeness × social_likelihood × relationship × proximity | 同左 (+ companion_boost) |
| ── 中身の方向 | system_prompt の中心問い + 受信メッセージ + memory | 同左 + 場所への perceive |
| 移動 | (なし、座席固定) | LLM 判断 (walk_toward 等 6種) |
| ── 移動の方向 | ─ | persona + memory + nearby_places + 中心問い から LLM 文脈判断 |
| 場所訪問の動機 | ─ | (構造的に弱い)。FW_TASK は initial_memory にあるが rolling buffer で消える |
| 思考 | LLM call が起きれば毎回 memory + reasoning 書き込み | 同左 (発話 phase + 行動 phase 両方で書き込み) |

### 致命的な構造的問題
1. **FW_TASK の弱さ**: initial_memory のみで rolling buffer から消える → 後半 step では「2社訪問」を忘れる → 達成率 0/10
2. **動機の薄さ**: 「特定の場所に行く理由」が persona に紐付いていない (interest_tag は memory に明示されるだけ)
3. **アンケートの忖度バイアス**: 数値が actual behavior と乖離。質的評価 (memory + messages の中身) を見ないと判定できない
4. **Phase A の同調圧力**: 全員が同じ GL2020 を見て同じ視点に収束

### 採用済みの改善
- **① host current_goal を動機型に書き換え** (smoke4) → host が学生から学ぼうとする ✅ q7 +2.75 改善

### 未採用 / 検討中
- **(α) FW_TASK を system_prompt に固定挿入** → 後半 step でも常に「2社訪問」を意識
- **(β) 未訪問 host の rule-based 注入** → 「[まだ訪問してない: A, B, C]」を毎 step memory に
- **(γ) 興味タグ × 企業マッチング** → tech 興味の子に「ソニー/NTT は技術系」を勧める
- **Phase A 多様性増強** → 生徒ごとに異なる focus facts を渡して同調を防ぐ

