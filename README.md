# LLM Multi-Agent 2D Simulation

## 概要

LLMエージェントの **創発性** と **自律性** を発現させることを目的とした 2 次元空間マルチエージェントシミュレーション。エージェントには生の数値データ（占有率、火災の強度・距離等）のみを提供し、行動指示や定性的評価（「危険」「快適」等）は一切与えない。エージェント自身が状況を解釈し、意思決定・コミュニケーションを通じて集団的な行動パターンが創発する様子を観察する。

**フィールドには複数の場所（デフォルトでは左側と右側にバーが 1 つずつ）が存在し、各場所は独立した収容上限（capacity）を持つ。シミュレーション途中で火災イベントも発生する。**

本リポジトリは Anthropic Claude API（Haiku 4.5）を LLM バックエンドに使用し、プロンプトキャッシング・並列化・最小ペルソナなど一連の最適化を実装している。詳細は [`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md) を参照。

### Fire Event（火事イベント）

シミュレーションの途中で、指定位置（またはランダムな位置）に複数の火事を発生させることができる。火事には以下の特徴がある:

- **知覚モデル B（距離依存）**: 火事の知覚半径（`radius`）内にいるエージェントのみが火事情報を直接受け取る。半径外のエージェントはプロンプトに火事情報が一切含まれず、他エージェントからのメッセージ経由でのみ間接的に知る
- **定量情報のみ**: エージェントに与えられるのは火事の位置、強度（0.0〜1.0）、半径、自分との距離のみ。「危険」「避難すべき」等の定性的記述は一切含まれない
- **時間不変**: 火事の強度・位置は発生後変化しない
- **発生前は無情報**: 火事が発生するステップより前のプロンプトには、火事に関するセクションは存在しない

この設計により、エージェントが定量情報のみからどのような行動（回避、情報伝達、無視等）を創発的に生み出すかを観察できる。

### ペルソナ

各エージェントには初期化時に最小ペルソナが割り当てられる:

- `name`（例: 田中健一）
- `age`
- `gender` (`male` / `female`)
- `occupation`（例: IT会社員、デザイナー、定年退職）
- `background`（例: 最近この街に引っ越してきた／この街に長く住んでいる）
- `speech_style`（例: 丁寧だが少し堅い／カジュアルでフレンドリー）

`config.yaml` の `agents.personas` で一部を明示的に指定でき、未指定のエージェントは `utils.py` 内のプールからランダム生成される。プロンプトには各エージェントの「WHO YOU ARE」セクションが含まれ、近隣エージェントの表示や受信メッセージの差出人も **名前ベース** になる（ID の露出はプロンプト上では避ける設計）。結果として、座標報告が中心だった会話が、自然な人間らしい対話に変わる。

## 必要な環境

- Python 3.8 以上
- Anthropic Claude API のアカウントと API キー
- 必要な Python パッケージ（`requirements.txt` 参照）
- （任意）FFmpeg — 動画生成を行う場合のみ

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

`.env.example` を `.env` にコピーし、Anthropic Claude API キーを設定:

```bash
cp .env.example .env
# エディタで .env を開き、ANTHROPIC_API_KEY を自分のキーに差し替える
```

`.env` は `.gitignore` に含まれているのでコミットされない。API キーは [Anthropic Console](https://console.anthropic.com/) で発行できる。

### 3. 設定ファイルの確認

`config.yaml` で主要なパラメータを確認・調整する:

- エージェント数 / ステップ数
- `llm.model`（デフォルト: `claude-haiku-4-5-20251001`）
- 並列度 `agents.parallel_workers`（Tier に応じて 10〜20 程度）
- ペルソナ定義

## 使用方法

```bash
# 仮想環境を有効化
source venv/bin/activate              # macOS/Linux
venv\Scripts\activate.bat             # Windows

# 基本実行
python main.py

# 可視化を有効化
python main.py --visualize

# フレームを保存
python main.py --save-frames

# カスタム設定を使用
python main.py --config custom_config.yaml
```

動作確認用に 3 エージェント × 5 ステップの軽量設定（`config_opt1_test.yaml`）も同梱している。

```bash
python main.py --config config_opt1_test.yaml
```

## 設定ファイル（config.yaml）

主要なパラメータ:

- **simulation**: シミュレーション設定
  - `duration`: シミュレーションステップ数
  - `half_space_size`: 空間の半分のサイズ（例: 25 → 座標範囲は -25 〜 +25）
  - `half_place_size`: 場所の半分のサイズのデフォルト（各場所の `half_size` が優先）

- **agents**: エージェント設定
  - `num_agents`: エージェント数
  - `communication_radius`: 通信半径
  - `memory_limit`: 保存する最大メモリ数（デフォルト: 20）
  - `memory_size`: LLM 推論に使用するメモリ数（デフォルト: 5）
  - `message_history_limit`: 保存する最大メッセージ数（デフォルト: 10）
  - `message_context_size`: LLM 推論に使用するメッセージ数（デフォルト: 3）
  - `skip_probability`: 各フェーズで LLM 呼び出しをランダムにスキップする確率（デフォルト: 0.2）
  - `parallel_workers`: 1 フェーズあたりの同時 LLM 呼び出し数（1 = 逐次実行、上限は Anthropic Tier 依存）
  - `personas`: オプショナルな明示ペルソナのリスト（未定義 ID はランダム生成）

- **places**: 場所設定（複数場所対応）
  各場所は以下の必須フィールドを持つ:
  - `name`: 場所名（例: `"left_bar"`, `"cafe"`）
  - `type`: 場所の種類（例: `"bar"`, `"cafe"`, `"library"`）
  - `center_x`, `center_y`: 場所の中心座標
  - `half_size`: 中心からの半サイズ（±half_size が場所範囲）
  - `capacity`: 収容上限（占有率の計算に使用、ハードリミットではない）

- **fires**: 火事イベント設定（複数火事対応）
  - `name`: 火事名
  - `start_step`: 発生ステップ
  - `intensity`: 強度（0.0〜1.0、定量値としてのみ伝達）
  - `radius`: 知覚半径
  - `center_x`, `center_y`: 位置（省略でランダム）

- **llm**: LLM 設定
  - `model`: Claude モデル名（デフォルト: `claude-haiku-4-5-20251001`）
  - `base_url`: Anthropic API エンドポイント
  - `temperature`: サンプリング温度
  - `max_tokens`: 最大出力トークン数（デフォルト: 600、truncation 検出時は 1500 に自動拡張して 1 回だけリトライ）

- **visualization / logging**: 出力設定

### 座標系

- **フィールド**: 原点 (0, 0) を中心に、`-half_space_size` 〜 `+half_space_size` の範囲
- **場所**: 各場所は独立した中心位置（`center_x`, `center_y`）を持ち、そこから `±half_size`（両端を含む）

**エージェントの知識**:
- エージェントは **すべての場所の位置情報** を知っている（プロンプトに含まれる）
- 場所の占有状況（エージェント数・収容上限・占有率）は、その場所内にいるエージェントのみが直接受け取る

## 出力

- `output/`: 可視化フレームと統計グラフ
  - 可視化では、エージェントの **性別を色**（男=青、女=赤）、**場所内/外をマーカー形状**（場所内=★、場所外=●）で表現
  - 火事発生後は **火事中心（赤三角）** と **知覚半径（破線円）** が描画される。`intensity` に応じた YlOrRd カラーマップで塗られ、カラーバー（0.0〜1.0, "Intensity of Fire"）も表示される
  - 統計グラフには「火事半径内エージェント数の時系列」サブプロットが追加される
- `output/messages.jsonl`: エージェント間メッセージ履歴
- `output/memory_reasoning.jsonl`: 各エージェントの記憶と推論ログ
- `output/report.html`: HTML レポート
- `output/animation.gif`: GIF アニメーション
- `simulation.log`: シミュレーションログ

## シミュレーション結果の可視化ツール

`visualization/` ディレクトリにビューアと動画生成スクリプトが含まれる。

### ブラウザビューア（viewer.html）

`visualization/viewer.html` をブラウザで開き、`output/` ディレクトリを選択するとインタラクティブに再生・閲覧できる。

| 操作 | 方法 |
|---|---|
| 再生 / 停止 | 「▶ 再生」 / `Space` |
| 前のステップ | 「⏮ 前へ」 / `←` |
| 次のステップ | 「⏭ 次へ」 / `→` |
| ステップジャンプ | スライダー |
| 再生速度調整 | 速度スライダー（0.5x〜3.0x） |

### 動画生成（generate_video.py）

`viewer.html` と同レイアウトで MP4 を生成するスクリプト。FFmpeg 必須。

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

```bash
python visualization/generate_video.py output/
python visualization/generate_video.py output/ -o result.mp4 --fps 20
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `data_dir`（必須） | — | シミュレーション結果ディレクトリ |
| `-o`, `--output` | `simulation.mp4` | 出力 MP4 ファイル名 |
| `--fps` | `10` | フレームレート |
| `--dpi` | `150` | 画像解像度 |

## LLM エージェント設計

### 基本設計

- 各エージェントは毎ステップ、**Message / Memory / Action** を LLM から生成し、同期的に行動
- 同一の LLM クライアントを全エージェントで共有し、差分は各自のペルソナ・Memory・相互作用履歴のみ
- プロンプトは「最適化タスク」を明示せず、状況説明 + 数値データ + 近傍メッセージ + 自己状態 + ペルソナを与える構成

### Action（行動）

4 方向の離散選択 + 滞在:
- `up` / `down` / `left` / `right`（±1 セル）
- `stay`（現在位置に滞在）

### Memory（記憶）

- LLM が出力した `memory` フィールドが次ステップの「Previous Memory」として自己フィードバック
- 内部状態が履歴依存で進化し、個性が創発する
- `memory_limit`: 保存する最大メモリ数（古いものから削除）
- `memory_size`: LLM 推論時に参照する直近のメモリ数

### 場所内限定情報

**各場所内のエージェントのみ** が以下の数値データを直接受け取る:
- 現在のエージェント数（Number of agents here）
- 収容上限（Capacity）
- 占有率（Occupancy rate = エージェント数 / 収容上限）

定性的な評価（快適・不快等）は一切含まれず、数値の解釈はエージェント自身に委ねられる。場所外のエージェントはこれらの情報を受け取らない（会話や推論で間接的に学ぶ）。

### コミュニケーション

- 通信半径内（デフォルト: 5 セル）のエージェント間でメッセージ交換が可能
- **同一領域条件**:
  - ✅ 同じ場所内のエージェント同士: 通信可能
  - ✅ 両方とも場所外のエージェント同士: 通信可能
  - ❌ 場所内のエージェント ↔ 場所外のエージェント: 通信不可
  - ❌ 異なる場所内のエージェント同士: 通信不可
- メッセージは **ブロードキャスト**（通信範囲内の全員に同じ内容が届く）
- `message_history_limit` / `message_context_size` で保存・参照数を調整

### シミュレーションステップの実行順序

各ステップは以下の順序で実行される:

0. **火事活性化チェック** — 各火事について `step >= start_step` なら発生
1. **Phase 1: メッセージ決定** — 全エージェントが LLM でメッセージを決定（並列実行）
2. **Phase 2: メッセージ送信** — 意思決定時点での近傍エージェントにメッセージを送信
3. **Phase 3: 行動決定** — 全エージェントが LLM で行動（move/stay）を決定（並列実行）
4. **Phase 4: 移動実行** — エージェントが決定した方向に移動

この順序により、メッセージは移動前の位置関係に基づいて送信される。

### LLM 出力形式

**メッセージ決定（Phase 1）**:
```json
{
    "message": "近傍エージェントへのメッセージ（任意、最大200語）",
    "reasoning": "メッセージ送信の理由"
}
```

**行動決定（Phase 3）**:
```json
{
    "action": "move" or "stay",
    "direction": "up" | "down" | "left" | "right",
    "memory": "次ステップのために記憶したいこと",
    "reasoning": "決定理由"
}
```

## 実装されている最適化

Anthropic Claude API を安定的かつ低コストで運用するため、以下 5 つの最適化を段階的に実装している。詳細は [`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md) を参照。

1. **プロンプトキャッシング** — system prompt に `cache_control: ephemeral` を付与
2. **`max_tokens` 削減 + truncation 検知リトライ** — デフォルト 600、JSON 不完全検出時は 1500 で 1 回だけリトライ
3. **ランダムスキップ** — 各フェーズで `skip_probability` の確率で呼び出しを省略
4. **ThreadPoolExecutor 並列化** — `parallel_workers` 個まで同時に LLM 呼び出し
5. **最小ペルソナと口調指示** — 名前・年齢・職業・背景・話し方を各エージェントに付与し、会話を名前ベースに

本番ラン（20 エージェント × 50 ステップ）における実測:
- コスト削減率: **61.5%**（キャッシュ無効と比較）
- 会話リアリティ: 座標報告中心 → 名前ベースの自然な対話
- 統計（占有率等）は最適化前と同系の傾向を維持

## ライセンス

GNU General Public License v3.0

詳細は [LICENSE.txt](LICENSE.txt) を参照。

## 謝辞

本リポジトリは上流のシミュレーション設計思想をベースに、LLM バックエンドを Anthropic Claude API に置き換え、コスト・実行時間・会話品質の最適化を加えたものである。
