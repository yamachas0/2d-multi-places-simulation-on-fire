# V2 emergency_boost 倍率検証

**対象**: `simulation.py` `should_speak` 内の `emergency_boost` 実装
**目的**: Phase 2 (JR運休イベント) の情報伝播設計は3倍前提。実装と実ログが仕様通り3倍になっているかの三点照合
**結論**: **設計 / 実装 / 実ログとも3倍で一致。修正不要。** Phase 1 レポートの「最大2.0」表記のみ誤り(下記 §4 参照)

---

## 1. コードレベル確認

### 該当箇所 `simulation.py:723`

```python
emergency_boost = 3.0 if self._fires_near(agent, multiplier=2.0) else 1.0
```

### 該当箇所 `simulation.py:739`

```python
p = min(1.0, p_raw * emergency_boost)
```

### 該当箇所 `simulation.py:683-691` (_fires_near 定義)

```python
def _fires_near(self, agent: Agent, multiplier: float = 2.0) -> bool:
    """True if any active fire is within `multiplier × radius` of the agent."""
    for fire in self.fire_states:
        if not fire.get('active'):
            continue
        dist = agent.distance_to(fire['position'])
        if dist <= fire['radius'] * multiplier:
            return True
    return False
```

### 読解

- `emergency_boost` の値は **3.0 または 1.0** の二択 (L723)
- `multiplier=2.0` は `_fires_near` に渡す **検知範囲倍率** (火災半径の2倍以内なら「近い」と判定)であり、boost 倍率ではない
- boost 適用後の cap は 1.0 (L739, `min(1.0, ...)`)

→ **実装は機能3仕様通り「3倍、cap 1.0」**

---

## 2. 実ログ検証

対象ファイル: `_urban_v2_out/should_speak_log.jsonl` (1,185行 / 3,819 candidate評価)

### boost 適用状況

| 項目 | 値 |
|---|---:|
| candidate評価総数 | 3,819 |
| boost 適用件数 (> 1.0) | **779** |
| boost 適用シェア | 20.4% |
| boost 固有値 | **{3.0: 779}** (全て 3.0 ちょうど) |
| boost 発動 step 数 | **21 steps** (step 40-60 連続) |

step 40 は `config_urban_v2.yaml` の火災イベント開始 step と一致。

### 実効倍率 (p_speak / p_raw 比)

boost 適用された779件について、以下の比を算出:

```
実効倍率 = c["p_speak"] / (c["talkativeness"] × c["social_likelihood"] × c["relationship"] × c["proximity_factor"])
```

| 統計 | 値 |
|---|---:|
| n (有効ratio件数) | 779 / 779 |
| min | **2.9877** |
| mean | **3.0003** |
| max | 3.0115 |
| 2.95 < ratio < 3.05 件数 | 779 (全件 unclipped) |
| < 2.95 件数 (1.0 cap 到達) | **0** |

min/max が 3.00 ピタリにならないのは `p_speak` がログ側で `round(p, 6)` されているため。cap (`min(1.0, ...)`) に頭打ちした候補は一件も無く、全候補で期待通り p_raw × 3 がそのまま反映されている。

---

## 3. 三点照合サマリ

| 観点 | 値 | 一致 |
|---|---|:---:|
| 設計 (機能3仕様) | × 3 倍、cap 1.0 | - |
| 実装 (`simulation.py:723, 739`) | × 3.0、`min(1.0, ...)` | ✅ |
| 実ログ実効 (779 cand) | mean = 3.0003 (min 2.988 / max 3.012) | ✅ |

**結論: 修正不要**。

Phase 2 の JR運休イベントは stranger 帯 (rel≈0.05) で平常 p_speak ≈ 0.003 を boost 3倍で 0.009 に引き上げ、運休半径内なら数ステップで発話が期待できる設計。この前提が成り立つことを確認した。

---

## 4. Phase 1 レポートの記述修正

`V2_PHASE1_URBAN_STRUCTURE.md` §4-1 の以下の記述は誤り:

> `emergency_boost` (火災近傍で**最大2.0**)

正しくは:

> `emergency_boost` は火災半径 × 2.0 以内のエージェントに対して **3.0 倍** (cap 1.0)

これは `_fires_near(multiplier=2.0)` の検知範囲倍率と boost 倍率を混同した文面ミスで、実装・動作はどちらも仕様通り。Phase 1 レポートのこの一行は後続コミットで訂正する。

---

## 5. サニティチェックの扱い

指示書 §1 の「2倍または別値であれば smoke test 再実行」は、実装が3倍で確認できたため不要。Phase 1 の `_urban_v2_out/` 実ラン自体が「火災あり (step40) / stranger ペアに boost 適用 / 発話観測」を既に含んでおり、追加の smoke test を走らせる必要はない。

実例 (`_urban_v2_out/messages.jsonl` より):
- step 43: 山田太郎 (id=6) → 石川裕子 (id=17, rel=0.07) 「おはようございます！何か煙が見えませんか？」
- step 46: 鈴木三郎 (id=8) が rel=0.06-0.78 の 6人に一斉発話 (boost + 近接群集)

stranger 帯 (rel ≈ 0.07) のペアが火災 step で発話している直接観測が既に取れており、「boost 3倍で情報伝播が成立する」Phase 2 前提は確定。
