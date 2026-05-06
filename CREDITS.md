# CREDITS — Acknowledgments

本リポジトリは派生著作物 (derivative work) です。
元コード・引用ライブラリ・参考にした資料などをここに明記します。

## 元コード (Base / Upstream)

- **作品名**: 2d-multi-places-simulation-on-fire （LLM Multi-Agent 2D Simulation, Phase 0: 2D 火事避難シミュ）
- **著者**: 兵頭博士（Singulab所属）
- **ライセンス**: GNU General Public License v3 (`LICENSE.txt` を参照)
- **本リポジトリでの扱い**:
  - 元コードのアーキテクチャ・コア概念 (LLMマルチエージェント、定量情報のみ提供、双方向同時発話排除、Jaccard類似フィルタ等) を派生・継承しています。
  - GPL v3 ライセンスをそのまま継承しています。

兵頭博士のオリジナル作品は **Singulabメンバー内での共有が原則** であり、本派生リポジトリの公開はハッカソン提出要件 (Public Repo URL 必須) を満たすためのものです。
派生著作物として GPL v3 を遵守し、本ファイル及び `README.md` 冒頭にて明示的に元著者へのクレジットを記載しています。

## 派生・追加コード (Derivative additions)

- **著者**: やまちゃそ (@yamachas0)
- **使用 AI コーディング支援**: [Claude Code](https://claude.com/claude-code) (Anthropic)
- **追加範囲**:
  - LLM バックエンド抽象化と Google Gemini クライアント (`llm_client_factory.py`, `gemini_client.py`)
  - 教室 AB シミュ (Phase A、座学60step)
  - 品川フィールドワーク シミュ (Phase B、屋外50step)
  - シミュ後アンケート (Phase C, `tools/run_survey.py`)
  - 統合レポートビルダ (`tools/render_v3_report.py`、他)
  - 3D ビューア (`visualization/viewer_3d.html`)
  - 品川 3D シーン構築データ (`scene_export_*.yaml`, `docs/shinagawa_*.yaml`)
  - その他 `tools/` 配下のビルド・分析スクリプト群

## 主要な依存ライブラリ

- [Anthropic Claude Python SDK](https://github.com/anthropics/anthropic-sdk-python) — Claude API クライアント
- [Google Generative AI Python SDK](https://github.com/google-gemini/generative-ai-python) — Gemini API クライアント
- [Playwright (Python)](https://playwright.dev/python/) — HTML→PDF レンダリング (`tools/html_to_pdf.py`)
- [Matplotlib](https://matplotlib.org/) — 可視化
- [PyYAML](https://pyyaml.org/) — config 読み書き

依存ライブラリの詳細は `requirements.txt` を参照してください。

## 参考にした外部資料

- 文部科学省「社会に開かれた教育課程」関連資料 — Phase A 座学の問い設計および考察セクション
- 国土地理院 / OpenStreetMap — 品川駅周辺の地理情報 (シーン構築の参照)
- 各企業の公開情報 — 品川駅周辺企業 (シミュ内ホスト) の業種設定の参考

> ※ 本シミュレーションのレポート内に登場する企業担当者の発話・意思決定はすべて LLM による生成であり、実在する企業・人物の見解とは一切関係ありません (レポート内に明記)。

## ライセンス

本リポジトリ全体は元コードを継承して GPL v3 のもとで配布されます。
詳細は `LICENSE.txt` を参照してください。
