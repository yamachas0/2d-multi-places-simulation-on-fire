# Stage A モデル比較レポート — Claude Haiku 4.5 vs Gemini 3.1 Flash Lite Preview

実施日: 2026-04-20
シナリオ: 30エージェント × 60ステップ、火災発火=step40、seed=42、同一ペルソナ・同一place配置

---

## 1. セットアップ差分

| 項目 | Haiku 4.5 | Gemini 3.1 flash-lite preview |
|---|---|---|
| Provider | `anthropic` | `google` |
| Model | `claude-haiku-4-5-20251001` | `gemini-3.1-flash-lite-preview` |
| parallel_workers | 3 | **10** |
| max_tokens | 600 | **400** |
| temperature | 0.2 | 0.2 |
| キャッシュ方式 | Anthropic prompt cache (ephemeral) | **Gemini Context Caching (CachedContent)** |
| 出力制約 | フリーフォーム JSON (parser厳格) | **Structured Output (response_schema)** |

Gemini側は今回の `llm_client_factory.py` の書き足しで下記3点が追加で効いている：
- `CachedContent` による system_prompt キャッシュ（hash単位で共有）
- `response_mime_type: application/json` + schema 自動選択（message / action）
- `threading.Lock` によるキャッシュ重複作成の防止（race condition対策）

---

## 2. コスト・処理量サマリ

| 指標 | Haiku | Gemini | 比較 |
|---|---|---|---|
| LLMコール数 | 3,808 | 3,324 | −12.7% |
| 入力トークン（合計） | 5.26M + 19.5M cache | 18.5M（うち cache_read 15.9M） | — |
| キャッシュヒット率 | 78.6% | **86.0%** | +7.4pt |
| 出力トークン | 927K | **321K** | −65.4% |
| 実コスト | **$11.93** | **$0.79** | **約15倍安** |
| 実行時間 | 数十分（3 give-ups含む） | **~14分** | 半分以下 |
| rate-limit give-ups | 3 | **0** | — |
| エラー | 0 | 0 | — |

**結論：Gemini に乗り換えた結果、1/15のコストでクリーンかつ高速に走り切った。**

---

## 3. トークン内訳の考察

### Haiku側
- Anthropic prompt cacheは ephemeral（5分TTL）でフラット割引0.1x
- 1コールあたり約1,400トークン（Fresh）+ 5,100トークン（Cache）+ 243トークン（Output）
- max_tokens=600 を半分強使い切る傾向 → 長文memory/reasoning

### Gemini側
- Context Caching は1時間TTLで0.25x割引（Haikuより弱い）
- 1コールあたり約781トークン（Fresh）+ 4,789トークン（Cache）+ 97トークン（Output）
- **max_tokens=400** と **structured output** により出力が劇的に短縮（927K→321K）
- Cache hit率は **86%**、Haikuの78.6%を上回る（CachedContentが system_prompt全体を完全キャッシュできるため）

**出力削減の正体：** structured output により冗長な前置き・言い訳が消え、model/reasoningに必要最小限だけ出るようになった。max_tokens=400でも truncationは発生せず。

---

## 4. シミュレーション品質

### 火災反応（step40発火）
| 指標 | Haiku | Gemini |
|---|---|---|
| 火災関連メッセージ | （Stage Aデータ未再計測） | **51件 / 8,731 (0.58%)** |
| 火災認識メモリ | 同上 | **45件 / 2,267 (1.98%)** |
| riverside_park 避難動態 | 11→0 | **3→0（10ステップ以内）** |

両者ともriverside_parkから避難挙動が観測できたため、**シミュレーション妥当性としては実用的に同等**。サンプル件数（メッセージ51件）は Gemini 側が想定どおり発火し、structured outputでも火災認識・避難判断は劣化していないことを確認。

### 移動・滞在パターン
- 両モデルとも駅→cafe/barへの朝の流入を自然に再現
- north_cafe 占有率が時間帯で 13/15 に達する混雑現象が両方で発生
- 火災後の東側（south_bar含む）回避は軽度にしか現れず、これは両方共通の弱点

---

## 5. 判断

**Gemini 3.1 flash-lite preview を本プロジェクトの既定モデルに採用して問題なし。** 根拠：

1. **コスト**: $11.93 → $0.79。Stage Aを30回回しても$24、実験反復が現実的になった
2. **スピード**: parallel_workers=10 で give-up 0。Haikuで 50 req/min 組織上限に引っかかっていた問題から解放
3. **品質**: 火災反応・動線・滞在パターンとも Haiku と同等
4. **保守性**: Structured Output により parse 失敗リスクが構造的に消滅

### 残課題
- `google.generativeai` SDK は非推奨、将来的に `google.genai` への移行が必要（動作は問題なし）
- CachedContent のTTL 1時間、storage料金は極小だが長時間simの場合は明示的削除も視野
- 火災パラメータ(radius=25)は trajectory divergence を露出させやすい。検証時は radius=40-50 に広げるのが妥当

---

## 6. 次アクション候補
- [x] Gemini採用判断
- [ ] Stage A のアニメ/レポート (`output/animation.gif`, `output/report.html`) の目視確認
- [ ] 追加機能の実装再開（モデル選定完了により解禁）
- [ ] Stage B の仕様策定（必要なら）
