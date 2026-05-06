# LLM Multi-Agent 2D Simulation — AUTOMATA HACKATHON 提出版

> **チーム**: えびやま（やまちゃそ／えびねこ）
> **作品名**: まちと企業がつくる学習環境は、子どもに何をもたらすのか
> **AUTOMATA HACKATHON 2026 提出物**

---

## 🎯 最終成果物 (審査員の方はここへ)

ハッカソン本案 (品川 高校生10人 / Phase B 50step / run #157) の最終成果物を [`FINAL_REPORT/`](FINAL_REPORT/) に同梱しています。

| ファイル | 内容 |
|---|---|
| [`FINAL_REPORT/v3_report_157_v21.pdf`](FINAL_REPORT/v3_report_157_v21.pdf) | **統合レポート PDF** — 提出用説明資料の本体 (72ページ、シミュ設計／結果／考察を網羅) |
| [`FINAL_REPORT/v3_report_157_v21.html`](FINAL_REPORT/v3_report_157_v21.html) | 統合レポート HTML 版 (PDF と同内容、ブラウザで開くと全体タイムラインが対話可能) |
| [`FINAL_REPORT/viewer_3d_bundled_157_v21.html`](FINAL_REPORT/viewer_3d_bundled_157_v21.html) | **3D ビューア** (品川駅周辺 160m×160m を 3D 化、step スライダーで全エージェントの軌跡を再生) |
| [`FINAL_REPORT/v3_logs_157_v21.html`](FINAL_REPORT/v3_logs_157_v21.html) | 全ステップログ (各 agent の思考・移動・発話・受信を step 単位で全て確認可能。バックデータ用) |

> シミュ結果の生データ (jsonl, simulation_data.json, 各 step の memory_reasoning) は `simulations/` 配下に出力されますが、`.gitignore` 済のため公開リポジトリには含めていません。再現したい方は下記「再現手順」を参照してください。

---

## 元コードについて (重要・Acknowledgments)

本リポジトリは **シンギュラボ所属の兵頭博士** による
LLM マルチエージェント 2D シミュレーション (`2d-multi-places-simulation-on-fire`)
を **派生** させたものです。

- **元コードの著者**: 兵頭博士（シンギュラボ）
- **元コードのライセンス**: GNU General Public License v3 (`LICENSE.txt` 参照)
- **本リポジトリの位置付け**: 派生著作物 (derivative work)。GPL v3 を継承して公開しています。

ベースとなる Phase 0 (2D 火事避難シミュ／Claude API ベース) を起点に、
やまちゃそ（@yamachas0）が Claude Code を用いて以下を大幅追加・改造しました:

- LLM バックエンドを Gemini に切替 (`llm_client_factory.py`)
- 教室 AB シミュ（Phase A: 座学 60step）
- 品川駅前フィールドワーク シミュ（Phase B: 屋外 50step）
- シミュ後アンケート (Phase C, `tools/run_survey.py`)
- 統合レポートビルダ (`tools/render_v3_report.py` 他)
- 3D ビューア (`visualization/viewer_3d.html`)
- 品川 3D シーンデータ (`scene_export_*.yaml`, `docs/shinagawa_*.yaml`)

詳細なクレジット一覧は [`CREDITS.md`](CREDITS.md) を参照してください。

---

## ハッカソン提出版の概要

**作品テーマ**: 「**教育 × まち × 企業**」のカケザンで生まれる、新しい学びの形の検証・シミュレーション。
学校の外にあるまち・企業・インフラ・公共空間を、子どもたちの学びを支える環境として捉え直し、
教育をまちに展開したときに、子どもたちの **視野・問い・主体感・参加スタイル** にどのような変化が
生まれるのかを LLM マルチエージェント・シミュレーションで検証します。

題材は **品川駅周辺での「企業協力型 学外教育プログラム」**。10人の生徒（高校生／社会人／小学生 など複数のペルソナバリエーション）と、品川駅周辺 12 社の企業担当者を LLM エージェント化し、以下の3フェーズの一連の学習プロセスとして再現します:

- **Phase A — 教室での座学** (10:00〜11:00 / 30 step): 品川の歴史・地形・整備方針などを学び、生徒は「品川の未来像」と「FW で確かめたいこと」を考える。
- **Phase B — 品川駅前フィールドワーク** (13:00〜14:40 / 50 step): 品川駅東西自由通路を起点に、生徒たちは 160m × 160m の屋外フィールドで企業担当者と対話。最低2社の訪問が課題。
- **Phase C — シミュ後アンケート**: 生徒・企業担当者それぞれが本プログラムについて 10段階評価および自由記述で回答。

LLM が「自分はどこに行きたいか」「誰に話しかけるか」「何を学んだか」をすべて自律決定し、
学習プロセス全体を通じた **視点の変化・気づき・主体感** を観察する設計です。

**シミュレーション全体の問い**:
> このまちのいいところや課題を様々な視点で探し、こうなったらいいと思う未来像を描いてください。

特に以下の3点に着目しています:

1. **多様な子どもたちの参加スタイルが成立する学習環境になっているか** — 学校適応度・国籍・身体特性・関心領域などの差を持つ生徒が、それぞれの特性に応じたスタイルで学びに加われているか。
2. **子どもたちの変化** — まち全体を学び場にしたとき、視野・視座・問い・行動意欲がどう変わるか。「自分もまちに関われる」という感覚が生まれるか。
3. **企業や地域と連携した学習環境にするために何が必要か** — 企業担当者との対話が、子どもの気づきや違和感を「問い」へと深める機会になるか。まちを学び場として機能させる設計要素は何か。

詳細な結果・考察・PDFレポートは Google フォーム経由で別途提出している `説明資料.pdf` (= `tools/render_v3_report.py` で生成された統合レポート) を参照してください。

---

## 本案で追加・改造した主な機能

### 1. 3 フェーズ構造の学習プロセス再現

教育プログラム全体を **教室での座学 → まちへ出てフィールドワーク → 振り返りアンケート** の 3 フェーズで実装。各フェーズの出力 (memory, 発話, 行動ログ) が次フェーズへの入力として引き継がれます。

- **Phase A 教室AB シミュ** (`tools/build_classroom_ab_config.py`): 30 step / 60分。教師不在で生徒同士が問いについて議論。条件 A (生徒のみ) / B (触媒人物=シンギュラボ代表 佐藤航陽さん 同席) を比較可能。
- **Phase B 品川FW シミュ** (`tools/build_shinagawa_field_config.py`): 50 step / 100分。160m × 160m の品川駅周辺フィールドを 3D シーンとして構築。生徒 10人 + 企業担当者 12社のホストエージェントが屋外で対話。
- **Phase C アンケート** (`tools/run_survey.py`): 各エージェントの全 step memory を観察者 LLM に渡し、「この人物がアンケートにどう答えるか」を推測 (本人が直接答えると忖度バイアスが入るため、観察者経由)。

### 2. 多様ペルソナ生成

`school_fit` (学校適応度) / `mobility` (車いす利用) / `nationality` (外国籍) / 興味領域 / 家庭背景 などの軸を加えた多様なペルソナを LLM で生成:

- `tools/generate_classroom_personas.py --variant {high|adult|elementary}`
- 高校生10人 / 社会人10人 / 小学生10人 の 3 variant をサポート (本案 = 高校生)
- 各 variant に外国人ルーツ参加者 3〜4名、車いす利用者、学校不適応 などを必ず含める

### 3. 統合レポートビルダ + 3D ビューア

- `tools/render_v3_report.py`: Phase A/B/C のシミュレーション結果から、エージェント挙動ルール解説・タイムライン・各人の言語化分析・Phase C アンケート集計・全体考察 (Gemini 生成) ・教育観点考察・実装観点考察 までを **1 本の HTML レポート** として出力 (本案では 72 ページの PDF として提出)。
- `tools/html_to_pdf.py`: Playwright 経由で HTML → PDF 化 (16:9 スライドサイズ、`@media print` でフォント縮小)。
- `visualization/viewer_3d.html`: 品川駅周辺 160m × 160m の 3D シーン。step スライダーで全エージェントの軌跡・発話バブルを再生可能。

### 4. LLM バックエンド抽象化 (Claude → Gemini 切替可能)

`llm_client_factory.py` でプロバイダ抽象化。Phase 0 (元コード) は Claude Haiku、Phase A〜C 本案は **Gemini 2.5 flash-lite** を採用 (Haiku 比でコスト 1/15、context cache hit 80%超、品質同等を実測検証)。

### 5. その他の挙動制御 (派生元コアを継承+調整)

派生元 (Phase 0) のコア (定量情報のみプロンプト提供 / 双方向同時発話排除 / Jaccard 4-gram 類似フィルタ / memory rolling buffer) を継承しつつ、本案では:

- **3 層行動モデル** (transit / observe / interact) — 屋外 FW 用に拡張
- **awareness 伝播** (path-1/2/3) — JR 運休や新幹線終電などのイベントが direct/notification/conversation 経路で agent 間に伝播する仕組み
- **constrained mode** — `scene_3d` を持つフィールド (品川など) で persona の `initial_place` を尊重した spawn
- **time-of-day guardrail** — 発話プロンプトの語彙を時間帯 (MORNING/EVENING/LATE NIGHT) で自動調整

詳細は `agent.py`, `simulation.py`, `navigation.py`, および `tools/build_*.py` 群を参照してください。

---

## 派生元 (Phase 0): 火事避難シミュレーション

兵頭博士の元コードである **Phase 0 — 2D 火事避難シミュレーション** は、本案 (Phase A〜C) のコア (LLM マルチエージェント / 定量情報のみ提供 / 双方向同時発話排除 / Jaccard類似フィルタ / memory rolling buffer / 並列化) のすべての基盤となっています。

Phase 0 の概要のみ記載: フィールド内に複数の場所 (バー等) があり、エージェントは火事の位置・強度・距離などの **数値データのみ** を受け取り、行動指示は一切与えられない状態で、回避・情報伝達・無視 等の行動を **創発的に生み出す** 設計。 `python main.py --config config.yaml` で動作。

> **Phase 0 のソース改変は最小限** に留め、本案 (Phase A/B/C) の追加機能はすべて `tools/build_*`, `tools/render_*`, `tools/generate_*` の新規スクリプト群と、`agent.py` / `simulation.py` / `navigation.py` の追加分岐として実装しています。

## 必要な環境

- Python 3.8 以上
- LLM API キー
  - **Phase 0** (火事避難シミュ): Anthropic Claude API キー
  - **Phase A〜C** (品川シミュ・本案): Google Gemini API キー (推奨: `gemini-2.5-flash-lite`)
  - `llm_client_factory.py` 経由でプロバイダ切替可能
- 必要な Python パッケージ（`requirements.txt` 参照）
- （任意）FFmpeg — 動画生成を行う場合のみ
- （任意）Playwright — `tools/html_to_pdf.py` で HTML レポートを PDF 化する場合のみ

## セットアップ

### 1. 仮想環境の作成

#### macOS / Linux

```bash
chmod +x setup_mac.sh
./setup_mac.sh

# または手動
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows

```cmd
setup_win.bat

REM または手動
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. API キーの設定

`.env.example` を `.env` にコピーし、使う API キーを設定:

```bash
cp .env.example .env
# エディタで .env を開き、必要なキーを設定する
```

- Phase 0 (火事避難シミュ) を動かす場合: `ANTHROPIC_API_KEY` ([Anthropic Console](https://console.anthropic.com/))
- Phase A〜C (品川シミュ・本案) を動かす場合: `GOOGLE_API_KEY` ([Google AI Studio](https://aistudio.google.com/app/apikey))

`.env` は `.gitignore` に含まれているのでコミットされない。

---

## 使用方法

### A. ハッカソン本案 (Phase A/B/C 品川シミュ) を回す

本案は **Phase A → Phase B → Phase C の 3 段** で動かします。各フェーズの出力を次フェーズの入力として渡します。

```bash
# === Phase A: 教室AB シミュ (高校生10人 / 30step / 60分) ===
# config 生成
python tools/build_classroom_ab_config.py --variant high --condition a
# 実行
python main.py --config config_classroom_ab_a.yaml

# 出力: simulations/<日時>_<id>_classroom_ab_a/
#   - fw_handoff.jsonl   (Phase B への引継ぎ: future_image / intent)
#   - field_questions.jsonl (Phase B での確かめたいこと)
#   - memory_reasoning.jsonl, messages.jsonl 他

# === Phase B: 品川FW シミュ (高校生10人 / 50step / 100分) ===
# Phase A の run_dir を渡して config 生成
python tools/build_shinagawa_field_config.py \
    --variant high \
    --classroom-run simulations/<Phase A の run_dir>
# 実行
python main.py --config config_shinagawa_field_high.yaml

# === Phase C: シミュ後アンケート ===
python tools/run_survey.py \
    --run-a simulations/<Phase A run_dir> \
    --run-b simulations/<Phase B run_dir>

# === 統合レポート (HTML + PDF) ===
python tools/render_v3_report.py \
    --run-a simulations/<Phase A run_dir> \
    --run-b simulations/<Phase B run_dir> \
    --suffix _v1
python tools/html_to_pdf.py simulations/<Phase B run_dir>/v3_report_<NN>_v1.html
```

社会人 / 小学生 variant も同じ流れで `--variant adult` / `--variant elementary` で起動可能 (本案 = 高校生)。

### B. Phase 0 (派生元の火事避難シミュ) を回す

```bash
python main.py --config config.yaml                # デフォルト
python main.py --config config_smoke.yaml          # 軽量 smoke
python main.py --config config_jr_disruption.yaml  # JR 運休イベント版
```

詳細パラメータは元コードの設計どおり (`config.yaml` 内のコメント参照)。

---

## 出力

- **`simulations/<日時>_<NN>_<name>/`** — 各 run の出力ディレクトリ (gitignore 済、ローカルにのみ生成):
  - `simulation_data.json` (各 step の position / state)
  - `messages.jsonl` (エージェント間メッセージ)
  - `memory_reasoning.jsonl` (各エージェントの memory + reasoning)
  - `actions.jsonl` (行動ログ)
  - `survey_responses.jsonl` (Phase C アンケート)
  - `config.yaml` (実行時の config スナップショット)
  - `sim.log` (LLM call 統計、cache hit 率等)
  - 生成した HTML レポート / PDF / 3D viewer (`tools/render_v3_report.py` / `html_to_pdf.py` 経由)
- **`FINAL_REPORT/`** — 本案 (run #157) の最終成果物を repo に同梱 (本ドキュメント冒頭参照)

---

## 主要な可視化ツール

| ファイル | 用途 |
|---|---|
| `visualization/viewer_3d.html` | **3D ビューア (本案メイン)**。品川駅周辺の屋外 3D シーンと全エージェントの軌跡・発話を step スライダーで再生 |
| `visualization/viewer_v2.html` | 2D ビューア (Phase 0 / 教室シミュ用)。場所・通信範囲・火事を 2D で表示 |
| `tools/bundle_viewer.py` | 上記 viewer + 全データを 1 ファイルの自己完結 HTML にバンドル (`viewer_3d_bundled_NNN.html`) |
| `tools/render_v3_report.py` | 統合レポート (HTML + 印刷対応 CSS)。本案の主要成果物 |
| `tools/html_to_pdf.py` | Playwright 経由で HTML → PDF (16:9 スライドサイズ、`@media print` でフォント縮小) |

---

## 主要な設計判断

- **Gemini 2.5 flash-lite を採用** (Phase A〜C): Haiku 比でコスト 1/15、品質同等を 30 agents × 60 steps の比較で実測検証。詳細は `FINAL_REPORT/` 配下の最終レポート内「やまちゃそ考察」セクションを参照
- **Phase A 固定 → B, C 独立** のアーキテクチャ方針: Phase A 出力を YAML として凍結することで、Phase B/C を独立に試行錯誤できるよう設計 (フルチェーンを毎回回すと A の LLM call が無駄に積み重なるため)
- **観察者 LLM 経由のアンケート (Phase C)**: 各 agent の memory を観察者 LLM に渡して「この人物がアンケートにどう答えるか」 を推測。本人に直接答えさせると忖度バイアスが入るため
- **属性マッピング厳守**: 多様ペルソナの属性 (学校不適応/車いす/外国籍) を考察 LLM プロンプト先頭で明示し、人物の取り違えを構造的に防ぐ
- **派生元 Phase 0 由来のコア (双方向同時発話排除 / Jaccard 4-gram 類似フィルタ / プロンプトキャッシング / 並列化)** を継承。Phase 0 の最適化詳細は [`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md) (元コード由来) 参照

---

## ライセンス

[GNU General Public License v3.0](LICENSE.txt) (派生元 Phase 0 を継承)。

## 謝辞

本リポジトリは **シンギュラボ所属の兵頭博士による LLM マルチエージェント 2D シミュレーション** を派生・発展させたものです。元コードの提供および設計思想に深く感謝します。詳細は [`CREDITS.md`](CREDITS.md) を参照。
