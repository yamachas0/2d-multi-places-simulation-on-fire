# Phase A/B エージェント挙動制御 — クイックスタート版からの変更点

**対象**: 品川 企業協力型 学外教育プログラム シミュ (本案 = 157)
**起点**: 当リポジトリの `quickstart_demo.py` 系のシンプルな振る舞い
**現在**: smoke23-31 系で固められた本案の挙動制御

---

## 1. 全体マップ (どの段階で何が変わったか)

```
quickstart 版                          現在の本案 (157)
────────────────                       ─────────────────────
1 step = decide_message + decide_action   →  Phase A は decide_action 省略 (stay 強制)、
                                              Phase B は両方 LLM call
全 agent が毎 step LLM 呼出                →  opener gate / skip_probability / Jaccard で
                                              不要な call は silent 化
nearby 全員に発話 broadcast                →  partner_id を 1人選んで targeted 配信、
                                              他 nearby は overhear
発話判定 = ランダム p_speak                →  partner 選択は p_speak 重み、silent 判定は
                                              「自分発しか動機ない step だけ確率 skip」
memory = rolling buffer のみ               →  rolling + 5step毎の archived_summary 圧縮 +
                                              system_prompt 固定領域に handoff intent を注入
場所コンテキスト = 静的                    →  perceive_pass / perceive_enter で 通過時 / 入場時に
                                              場所固有の文脈を memory に動的注入
host (受入担当) = 居るだけ                 →  動機型 current_goal +「学生 nearby なら必ず opener」 +
                                              入場時の opener 強制 + 終盤は配置に固定 (smoke31以降)
```

---

## 2. Phase A (教室座学) で変えたこと

### 2.1 行動の単純化
- **decide_action を skip** (`skip_decision_prompt: true`) — Phase A は教室固定で移動が無いので、毎 step `action_type=stay` で stub 返却。LLM call は decide_message のみ
- **memory / reasoning は decide_message と一緒に書く** (= 1 call/step に圧縮)

### 2.2 発話判定
- **opener gate**: 自分宛の受信が無い step は確率で skip → 「自分から話しかけるしか動機がない step」だけ LLM call を burn しない (smoke16)
- **opener 確率 2x ブースト**: 「最初は様子見で黙る」が過剰だったので best_prob × 2 (smoke17)
- **silent はランダム判定でなく LLM 任せ** + **Jaccard 4-gram ≥ 0.50 で類似発話を後フィルタ silent化** (smoke14-15)
- **双方向同時発話排除**: A→B と B→A が同 step で起きたら id 大きい方を silent set に → turn-take 自然化

### 2.3 議論を引き出す
- **集団タスク化 prompt**: 「教室で集まったメンバーで話し合う」 文脈をシーン text として注入 (#64)
- **同じ人に同じこと禁止**: 「あなたは {相手} に既に〇〇を言いました。同じテーマを繰り返さない」を user_prompt に動的に貼る (#72)
- **Phase A 補足 prompt**: 「あなたは集まったメンバーで議論する立場、自分の意見だけ繰り返さず他人の話を引き出す」(#64)
- **talkativeness を 0.55 → 0.66 → 0.76** (3 段階で引き上げ、後半の発話失速対策、効果は限定的)

### 2.4 引継ぎ (handoff) 抽出
- Phase A 終了時に各生徒の memory + 発話履歴を Gemini に渡し、**`fw_handoff.jsonl` (intent / future_image / key_memories) と `field_questions.jsonl` (FWで確かめたいこと 3問 + one_liner)** を抽出
- これが Phase B persona に焼き込まれて、座学からの問題意識が FW に持ち越される

---

## 3. Phase B (FW) で変えたこと

### 3.1 移動・行動
- **decide_action 復活** (`skip_decision_prompt: false`、`minimal_prompt_mode: false`) — walk_toward / approach / enter / wander / walk_along を LLM が選択
- **navigation の歩道制約撤廃** — 国道15号など道路全幅を歩行可に (旧版は歩道側だけしか歩けず詰まった)
- **place の polygon 内に入れば自動 enter** (= ground truth は LLM の action_type ではなく座標、`current_place_after` で記録)
- **東西自由通路スタート + grid 配置**: 12人を 12×1 grid で東西自由通路 polygon 内に密集スポーン (重なりなし、はみ出しなし)
- **車いす persona は移動速度 1/2** (movement_base_cells // 2)

### 3.2 発話判定 (Phase A と共通の機構)
- 同じく **opener gate / Jaccard 後フィルタ / 双方向同時排除 / 同テーマ禁止**
- **host opener gate を強制 ON**: host (受入担当) は学生が nearby なら必ず LLM call → 学生は遠慮しがちでも host から声掛けが入る (smoke23)
- **不適応の opener はゼロ**: school_fit=不適応 + 受信トリガー無し なら partner=None で確実に silent (smoke23)

### 3.3 system_prompt の構造化 (cache 領域に固定で持つもの)
- **WHO YOU ARE**: persona block (毎step同じ → cache hit)
- **goal block**: 中心問い (1文)
- **fw_task block**: 「FW中、最低 2社 (assigned) を訪ねる」 を **常時注入** (smoke21、rolling buffer から消えない)
- **premise block**: 13:00開始 / 集合場所 = 東西自由通路 / 終了時は集合場所に戻る (今の最終仕様)
- **handoff block**: Phase A の intent / future_image / key_memories / field_questions / one_liner を **system_prompt に固定挿入** (rolling buffer 押し出しの影響を受けない)
- **多様性ヒント**: school_fit=不適応 / gender=other / nationality / mobility=wheelchair の persona に対する対人傾向ヒントを system_prompt に固定挿入 (smoke22)
- **age block** (小学生のみ): 10歳の語彙レベル維持 / 大人語禁止リスト を固定挿入

### 3.4 user_prompt (毎step変動、cache外)
- **WORKING STATE block**: 動的に「残り step / 訪問進捗 / 未訪問の host / まだ確かめていない問い」を出す (smoke7-③)
- **per_partner_history**: 「{相手} に過去言ったこと top 3」 を可視化 → 同テーマ禁止 prompt と組合せ (smoke7-②)
- **cooldown step1 ループ検知警告**: 同 action を連続したら警告を user_prompt に挿入 (smoke8)
- **nearby_agents context 簡素化**: 旧 全項目 → 新 id/gender/distance/in_place の最小限に (cost 削減)
- **訪問場所を memory に rule-based で残す**: LLM に書かせず、入場時に `[{place_name} に到着]` を機械挿入

### 3.5 host 側
- **host current_goal の動機型化**: 「自社拠点に居て、学生に自社・まちの話を語るが、それと同等に **学生から見えるまちの違和感を引き出したい**」(smoke6 / #61)
- **host 不動強制**: simulation.py Phase 3 で `is_host=true` のとき action_type=stay を強制 (LLM call スキップ。smoke31以降。157は前)
- **assigned_hosts の round-robin 焼き**: 各学生に必ず訪問する 2社の host を build 時に均等割り
- **host の配置を「最寄り道路の歩道側 / 東西自由通路に最も近い角」に補正** (#43, smoke23)

### 3.6 終盤の戻り (smoke31以降の本案変更とは別系統)
- **WORKING STATE で残り step を毎step表示**
- **残り 20% を切ったら「集合場所への戻りも視野に」 を緩く表示** (案A)
- **残り 5 step を切ったら rule-based で walk_toward 東西自由通路 を強制** (decide_action の LLM 出力を上書き)

### 3.7 Phase C アンケート
- agent 自身に答えさせるのではなく、**観察者として「この生徒はどう答えるか」 を推測**させる方式 (#33, smoke14-④ evidence-bound)
- 数値設問は **evidence (具体的事実) 必須** で、ground_truth (rule-based 集計の対話 host 数 / 入場場所) を user_prompt 冒頭に貼って盛りを防止
- メタ語 ('行動ログ' '記録' 'memory' 'シミュレーション') 禁止を system_prompt で明示 (= 一人称で書く)

---

## 4. クイックスタートに無くて、本案にあるもの (機構レベル)

| 機構 | 役割 | 効いた局面 |
|---|---|---|
| **handoff (Phase A→B)** | 座学の問題意識を FW に持ち越し | FW 終盤 step まで「自分は何を見たかったか」が消えない |
| **system_prompt 固定領域** | cache 領域に persona / goal / fw_task / handoff / 多様性 / age を固定 | rolling buffer 押し出しの影響を受けず後半 step も意識継続 |
| **opener gate** | 自分発しか動機ない step だけ確率skip | LLM call 数 ~30% 削減、対話の自然性は維持 |
| **Jaccard 4-gram フィルタ** | 過去発話と類似なら silent | 「同じこと繰り返す」 を後段でカット |
| **双方向同時発話排除** | A→B / B→A が同 step → id小が話す | turn-take 自然化、片方が黙る |
| **per_partner_history** | 「相手に既に言ったこと」 を user_prompt に動的注入 | 同じ人に同じ話題で詰まらない |
| **memory 圧縮 (5step毎)** | raw 5件を 1 文要約して archive | rolling buffer サイズを保ちつつ長期記憶 |
| **WORKING STATE block** | 残り step / 進捗 / 未訪問 / 未解決問い を動的に表示 | 後半でも「何が残ってるか」 を agent が見える |
| **evidence-bound survey** | rule-based 集計をアンケート時に prompt 冒頭に貼る | 「実際は対話してないのに『深く話せた』」 を防止 |
| **cooldown step1 ループ検知** | 同 action 連続したら警告 prompt | LLM が「動かなくなった」 ときに別 strategy を促す |
| **多様性ヒント (school_fit / nat / mobility / gender)** | 各特性に薄いヒント | 「全員が同じ深さで参加」 ではない多様な参加スタイル |

---

## 5. 「変えたほうがよかった」 に消えていった案 (副作用で revert したもの)

- **moltbook風 global channel** (smoke9): 全 agent が共有チャンネルを読む実験。情報量過多で逆に発話が減ったので revert
- **chat session 完全 thread化** (smoke11): 各 agent に Gemini chat session を持たせる実験。コンテキスト保持の利点はあるが、LLM call フォーマットが異なり整合性が崩れた
- **Example B「会話中stay」誘導** → smoke31 で削除 (regression 起こしたため revert で復活)
- **「強制 walk_toward 帰路ルール」 を premise block (cache領域) に書く** → 序盤から「対話控えて移動」 と LLM が学習し regression、案A で削除

---

## 6. 効果のまとめ (本案 157 で最終的に効いてる挙動)

- 全員 1社以上 host と対話: **10/10**
- assigned 2社両方達成: **5/10**
- 物理 polygon 入場: 7/10 (うち 2/10 は両方、5/10 は片方)
- 後半 step まで自分の問題意識を保持: handoff intent と field_questions が memory + system_prompt 両系列に居続ける
- 多様性ペルソナ別の挙動差が見える (= 全員均一参加ではない)
- 「動かなくなる」 case が一部残る (車いず辻、不適応宮原、吉田 (じくね))。これは構造的・現実的な課題として残し、レポートで考察対象に
