"""Phase C: 企業協力型 学外教育プログラムシミュ後のアンケートを実行する。

各 agent に Phase A (座学) + Phase B (FW) の memory を context として渡し、
設問 10問 (生徒6 / 企業 host 4) に答えさせる。Gemini 並列実行。

Usage:
  python tools/run_survey.py --run-a <phaseA_run> --run-b <phaseB_run> [--out <jsonl>]
"""
from __future__ import annotations
import argparse, datetime, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402


def _load_memory(run_dir: Path, agent_id: int) -> list[str]:
    """memory_reasoning.jsonl から指定 agent の (step, memory, reasoning) を時系列で集める。"""
    p = run_dir / "memory_reasoning.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("id") != agent_id:
            continue
        mem = (d.get("memory") or "").strip()
        rea = (d.get("reasoning") or "").strip()
        if mem or rea:
            out.append(f"[step{d.get('step')}] mem: {mem} | reason: {rea[:240]}")
    return out


def _load_messages(run_dir: Path, agent_id: int) -> tuple[list[dict], list[dict]]:
    """指定 agent の sent / recv を時系列で。"""
    p = run_dir / "messages.jsonl"
    if not p.exists():
        return [], []
    sent = []
    recv = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        if m.get("from") == agent_id:
            sent.append(m)
        if m.get("to") == agent_id:
            recv.append(m)
    return sent, recv


def _build_ground_truth(persona: dict, all_personas: list[dict],
                         sent_b: list[dict], recv_b: list[dict],
                         mem_b_lines: list[str]) -> dict:
    """rule-based に「実際にこの agent がやったこと」を集計する。
    survey の盛り防止 (evidence-bound) 用。LLM 生成ではなく、ログから機械的に取り出す。"""
    is_host = bool(persona.get("is_host")) or str(persona.get("axis_id", "")).startswith("Host_")
    host_id_set = {p["id"] for p in all_personas if p.get("is_host") or str(p.get("axis_id", "")).startswith("Host_")}
    student_id_set = {p["id"] for p in all_personas if p.get("id") not in host_id_set}
    name_by_id = {p.get("id"): p.get("name", f"#{p.get('id')}") for p in all_personas}

    if is_host:
        # host 視点: 自分と話した学生
        partner_ids = set()
        for m in sent_b:
            tid = m.get("to")
            if tid in student_id_set:
                partner_ids.add(tid)
        for m in recv_b:
            fid = m.get("from")
            if fid in student_id_set:
                partner_ids.add(fid)
        talked_with = sorted(name_by_id.get(i, f"#{i}") for i in partner_ids)
        return {
            "role": "host",
            "talked_with_students": talked_with,
            "talked_count": len(talked_with),
            "n_sent": len(sent_b),
            "n_recv": len(recv_b),
        }
    else:
        # student 視点: 自分と話した host
        partner_ids = set()
        for m in sent_b:
            tid = m.get("to")
            if tid in host_id_set:
                partner_ids.add(tid)
        for m in recv_b:
            fid = m.get("from")
            if fid in host_id_set:
                partner_ids.add(fid)
        talked_hosts = sorted(name_by_id.get(i, f"#{i}") for i in partner_ids)
        # visited_places: memory_reasoning の memory 部分から「訪問履歴」「入場済み」 line を拾う
        # (rule-based 訪問記録は agent.py で memory に書き戻している)
        entered = set()
        passed = set()
        for line in mem_b_lines:
            if "入場済み:" in line:
                # e.g. "[訪問履歴] 入場済み: 京急改札, 港南口広場 / 通過済み(未入場): ..."
                tail = line.split("入場済み:", 1)[1]
                tail = tail.split(" / ", 1)[0].split("通過済み", 1)[0]
                for nm in tail.split(","):
                    nm = nm.strip()
                    if nm:
                        entered.add(nm)
            if "通過済み" in line:
                tail = line.split("通過済み", 1)[1]
                tail = tail.split(":", 1)[1] if ":" in tail else ""
                for nm in tail.split(","):
                    nm = nm.strip().rstrip(")").rstrip("】")
                    if nm and nm != "未入場":
                        passed.add(nm)
        return {
            "role": "student",
            "talked_hosts": talked_hosts,
            "talked_count": len(talked_hosts),
            "entered_places": sorted(entered),
            "passed_places": sorted(passed - entered),
            "n_sent": len(sent_b),
            "n_recv": len(recv_b),
        }


def _format_ground_truth_block(gt: dict) -> str:
    """ground_truth dict を user_prompt 冒頭に貼るテキストブロックに整形。"""
    lines = ["## 実際の行動ログ (改ざん不可・rule-based 集計)"]
    if gt["role"] == "student":
        if gt["talked_hosts"]:
            lines.append(f"- 実際に対話した企業担当者 ({gt['talked_count']}社): {', '.join(gt['talked_hosts'])}")
        else:
            lines.append("- 実際に対話した企業担当者: **0社 (1人とも対話していない)**")
        if gt["entered_places"]:
            lines.append(f"- 入場した場所: {', '.join(gt['entered_places'])}")
        else:
            lines.append("- 入場した場所: なし")
        if gt["passed_places"]:
            lines.append(f"- 通過した(未入場の)場所: {', '.join(gt['passed_places'])}")
        lines.append(f"- FW中の総発話: {gt['n_sent']}件 / 受信: {gt['n_recv']}件")
    else:
        if gt["talked_with_students"]:
            lines.append(f"- 実際に対話した学生 ({gt['talked_count']}人): {', '.join(gt['talked_with_students'])}")
        else:
            lines.append("- 実際に対話した学生: **0人 (1人とも対話できなかった)**")
        lines.append(f"- FW中の総発話: {gt['n_sent']}件 / 受信: {gt['n_recv']}件")
    lines.append("")
    lines.append("**重要 (絶対守る)**: 上記は run ログから機械的に集計した事実であり、改ざん不可。")
    lines.append("- 上のリストに**含まれていない人物・場所を「話した」「訪れた」「対話した」と書くのは禁止**。")
    lines.append("- 数値設問の点数は、上の事実から自然に出る値を選ぶ。例: 学生の場合 talked_hosts=0社なら、")
    lines.append("  「視野が広がった」「街に関われると思えた」設問の点数は通常 1〜3 になる (深い体験がない)。")
    lines.append("- 自由記述でも、上のリストにない場所・人物を体験談として書くのは禁止。")
    lines.append("- evidence 欄では、上のリストに載っている事実 (場所名・対話相手の名前・実際に発したこと) を引用する。")
    return "\n".join(lines)


STUDENT_QUESTIONS = """あなた (生徒) は以下の **6問すべて** に答えてください。
数値設問はいずれも **10段階評価 (1-10 の整数)**。両端の意味は各問に記載。

問1 (10段階評価, 1=印象悪化 / 5=変わらず / 10=大幅に向上): 座学時と比べて、品川という街の印象は変わりましたか？
問2 (10段階評価, 1=まったくない / 10=強くある): 自分の視野が広がった実感はありますか？
問3 (10段階評価, 1=まったく思えない / 10=強く思える): 街に「自分も関われる」と思えた度合いは？
問4 (自由記述): 座学で描いた未来像と、FW (フィールドワーク) で歩いて見えた現実は、どこが違いましたか？
問5 (自由記述): 教室での学びと街での学びの違いを、自分の言葉で書いてください。
問6 (自由記述): この教育プログラムの **良かった点 / 悪かった点・改善希望** を一つずつ書いてください。
"""

HOST_QUESTIONS = """あなた (企業 host = 受入担当) は以下の **4問すべて** に答えてください。
数値設問はいずれも **10段階評価 (1-10 の整数)**。両端の意味は各問に記載。

問7 (10段階評価, 1=まったく響かなかった / 10=深く議論できた): 受け入れた学生たちとの対話の手応えは？
問8 (10段階評価, 1=まったく価値なし / 10=ぜひ継続したい): このプログラムは、自社にとって価値ある取り組みでしたか？
問9 (自由記述): 学生を受け入れてみて気づいた、街と学校の連携の **良かった点** を書いてください。
問10 (自由記述): 改善してほしい点・運営上の懸念があれば書いてください。
"""

SYSTEM_PROMPT_STUDENT = (
    "あなたは観察者です。以下の人物 (生徒) のシミュレーション体験を読み、"
    "**この生徒がアンケートに正直に答えるとしたら、どう答えるかを推測してください**。\n\n"
    "**重要 1: user_prompt の冒頭に「実際の行動ログ (改ざん不可)」が貼られています。**"
    "これは run ログから rule-based に集計された事実であり、改ざん不可です。"
    "ここに含まれない人物・場所を「話した」「訪れた」「対話した」と書くことは禁止。"
    "数値設問の点数も、この事実から自然に出る値を選んでください。"
    "talked_hosts が 0社なら「視野が広がった」「街に関われると思えた」は通常 1〜3 です。\n\n"
    "**重要 2: 評価は甘めにせず、actual な体験 (どこを訪れたか / 誰と話したか / 何を memory に書いたか) "
    "から自然に出る数値を選んでください。体験が薄ければ低い数値、深ければ高い数値、が原則です。**\n\n"
    "**応答ルール (絶対守る):**\n"
    "1. 出力は **JSON オブジェクト 1つ** のみ。前置き・コメント・コードブロック禁止。\n"
    "2. 値はすべて **日本語**。英語は固有名詞のみ。\n"
    "3. 数値設問 (q1, q2, q3) の値は **1〜10 の整数**。\n"
    "4. 各数値設問には **q1_evidence** のような対応する evidence フィールドで、"
    "その点数の根拠となる **具体的な事実** (場所名・誰の発話・特定の気づき等) を 1-2 文で記述する。"
    "evidence が抽象的 (「いろいろ気づいた」等) や空の場合、その回答は信用できないと判定される。\n"
    "5. 自由記述 (q4, q5, q6) は 1-3 文の自然文 (40-200字目安)。「ご回答します」のような前置き禁止。\n"
    "6. **メタ語禁止 (絶対)**: 回答文中に「行動ログ」「ログ」「記録」「memory」「シミュレーション」"
    "「観察者として」「実際の行動ログによると」「データ上は」のような、シミュレーション側の仕組みを示す語を **一切** 使わない。"
    "あなたはその生徒本人として、一人称で「〜と聞いた」「〜を見た」「〜を訪ねた」「〜と感じた」のように書く。"
    "対話相手の名前・場所名は具体的に書いてよい (むしろ書くべき)。\n\n"
    "出力フォーマット:\n"
    "{\n"
    '  "q1": <1-10>, "q1_evidence": "具体的な事実1-2文",\n'
    '  "q2": <1-10>, "q2_evidence": "...",\n'
    '  "q3": <1-10>, "q3_evidence": "...",\n'
    '  "q4": "...", "q5": "...", "q6": "..."\n'
    "}\n\n"
    "出力の最初の文字は必ず `{`、最後の文字は `}`。"
)

SYSTEM_PROMPT_HOST = (
    "あなたは観察者です。以下の人物 (企業の受入担当) のシミュレーション体験を読み、"
    "**この担当者がアンケートに正直に答えるとしたら、どう答えるかを推測してください**。\n\n"
    "**重要 1: user_prompt の冒頭に「実際の行動ログ (改ざん不可)」が貼られています。**"
    "これは run ログから rule-based に集計された事実であり、改ざん不可です。"
    "ここに含まれない学生を「対話した」と書くことは禁止。talked_with_students が 0人なら、"
    "対話の手応え・プログラムの価値の点数は通常 1〜3 です (実体験がない)。\n\n"
    "**重要 2: 評価は甘めにせず、actual な体験 (実際に何人の学生と会話できたか / どんな対話だったか / "
    "memory に何が残ったか) から自然に出る数値を選んでください。学生がほぼ来なければ低い数値、"
    "深い対話があれば高い数値、が原則です。**\n\n"
    "**応答ルール (絶対守る):**\n"
    "1. 出力は **JSON オブジェクト 1つ** のみ。前置き・コメント・コードブロック禁止。\n"
    "2. 値はすべて **日本語**。英語は固有名詞のみ。\n"
    "3. 数値設問 (q7, q8) の値は **1〜10 の整数**。\n"
    "4. 各数値設問には **q7_evidence** のような対応する evidence フィールドで、"
    "その点数の根拠となる **具体的な事実** (どの学生と / 何を話したか / どんな反応だったか等) を 1-2 文で記述する。\n"
    "5. 自由記述 (q9, q10) は 1-3 文の自然文 (40-200字目安)。「ご回答します」のような前置き禁止。\n"
    "6. **メタ語禁止 (絶対)**: 回答文中に「行動ログ」「ログ」「記録」「memory」「シミュレーション」"
    "「観察者として」「実際の行動ログによると」「データ上は」のような、シミュレーション側の仕組みを示す語を **一切** 使わない。"
    "あなたはその担当者本人として、一人称で「〜と話した」「〜が訪ねてきた」「〜と感じた」のように書く。"
    "対話相手の名前は具体的に書いてよい (むしろ書くべき)。\n\n"
    "出力フォーマット:\n"
    "{\n"
    '  "q7": <1-10>, "q7_evidence": "具体的な事実1-2文",\n'
    '  "q8": <1-10>, "q8_evidence": "...",\n'
    '  "q9": "...", "q10": "..."\n'
    "}\n\n"
    "出力の最初の文字は必ず `{`、最後の文字は `}`。"
)


def _build_user_prompt(persona: dict, role: str, mem_a: list[str], mem_b: list[str],
                       sent_b: list[dict], recv_b: list[dict],
                       ground_truth_block: str = "") -> str:
    name = persona.get("name", "?")
    age = persona.get("age", "?")
    occ = persona.get("occupation", "")
    role_label = "生徒" if role == "student" else "企業 host (受入担当)"

    bg_block = ""
    if persona.get("background"):
        bg_block = f"\n## あなたの背景\n{(persona.get('background') or '').strip()[:500]}"

    mem_a_block = "\n".join(mem_a[:30]) or "(座学の記録なし)"
    mem_b_block = "\n".join(mem_b[:30]) or "(FWの記録なし)"
    sent_block = "\n".join(
        f"[step{m.get('step')}→{m.get('to_name', '')}] {(m.get('message') or '')[:200]}"
        for m in sent_b[:20]
    ) or "(FW中の発話なし)"
    recv_block = "\n".join(
        f"[step{m.get('step')} from {m.get('from_name', '')}] {(m.get('message') or '')[:200]}"
        for m in recv_b[:20]
    ) or "(FW中の受信なし)"

    questions = STUDENT_QUESTIONS if role == "student" else HOST_QUESTIONS
    out_keys_hint = "q1〜q6 (q1/q2/q3 には evidence ペア)" if role == "student" else "q7〜q10 (q7/q8 には evidence ペア)"

    gt_section = f"{ground_truth_block}\n\n" if ground_truth_block else ""

    return (
        f"{gt_section}"
        f"## 対象人物\n名前: {name}, 年齢: {age}, 役: {role_label}\n職業: {occ}\n{bg_block}\n\n"
        f"## Phase A (座学) でこの人物が残した思考記録\n{mem_a_block}\n\n"
        f"## Phase B (FW) でこの人物が残した思考記録\n{mem_b_block}\n\n"
        f"## Phase B でこの人物が発した発話\n{sent_block}\n\n"
        f"## Phase B でこの人物が受信した発話\n{recv_block}\n\n"
        f"## アンケート設問\n{questions}\n\n"
        f"上記すべてを観察し、この人物が正直にアンケートに答えるとしたらどう答えるかを推測して、"
        f"JSON で {out_keys_hint} のキーを出力してください。"
        f"\n\n**再掲**: 冒頭の「実際の行動ログ」に書かれていない人物・場所を体験として書くのは禁止です。"
        f"talked_count=0 等で実際の体験が薄い場合、点数は素直に低くしてください (盛らない)。"
    )


def _parse_json_response(text: str) -> dict:
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="Phase A (座学) run dir")
    ap.add_argument("--run-b", required=True, help="Phase B (FW) run dir")
    ap.add_argument("--out", default=None,
                    help="出力 jsonl path (default: <run_b>/survey_responses.jsonl)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    cfg_b = yaml.safe_load((run_b / "config.yaml").read_text(encoding="utf-8"))
    personas = cfg_b["agents"]["personas"]
    print(f"[info] Phase B personas: {len(personas)}")

    # 役判定: is_host or axis_id が Host_ で始まるなら host、それ以外は student
    def role_of(p):
        if p.get("is_host"):
            return "host"
        if str(p.get("axis_id", "")).startswith("Host_"):
            return "host"
        return "student"

    client = GeminiClient(enable_structured_output=False, enable_cache=False)

    def survey_one(p):
        agent_id = p.get("id")
        role = role_of(p)
        # Phase A は host にとって「座学不参加」なので空
        mem_a = _load_memory(run_a, agent_id) if role == "student" else []
        mem_b = _load_memory(run_b, agent_id)
        sent_b, recv_b = _load_messages(run_b, agent_id)
        # 盛り防止: rule-based 集計の ground_truth を作って prompt 冒頭に貼る
        gt = _build_ground_truth(p, personas, sent_b, recv_b, mem_b)
        gt_block = _format_ground_truth_block(gt)
        user = _build_user_prompt(p, role, mem_a, mem_b, sent_b, recv_b, gt_block)
        sys_prompt = SYSTEM_PROMPT_STUDENT if role == "student" else SYSTEM_PROMPT_HOST
        try:
            text = client.generate(sys_prompt, user, temperature=0.5, max_tokens=900)
        except Exception as e:
            return {
                "id": agent_id, "name": p.get("name"), "role": role,
                "error": str(e), "responses": {}, "ground_truth": gt,
            }
        j = _parse_json_response(text)
        return {
            "id": agent_id, "name": p.get("name"), "role": role,
            "axis_id": p.get("axis_id"),
            "responses": j,
            "ground_truth": gt,
        }

    print(f"[info] running survey ({args.workers} parallel) ...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(survey_one, p) for p in personas]
        for f in as_completed(futures):
            r = f.result()
            tag = "ok" if r.get("responses") else "FAIL"
            print(f"  [{tag}] {r.get('name')} ({r.get('role')}) -> q-keys={list(r.get('responses', {}).keys())}")
            results.append(r)
    dur = int(time.time() - t0)
    # id 順にソート
    results.sort(key=lambda r: r.get("id", 999))

    # 互換: 全件まとめて 1 つの jsonl は従来どおり書く
    out_path = Path(args.out) if args.out else (run_b / "survey_responses.jsonl")
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    print(f"\n[ok] wrote {out_path} ({len(results)} responses, {dur}s)")

    # 学生 / 企業 host を別ファイルにも書き出し (後段レポートで使う)
    student_results = [r for r in results if r.get("role") == "student"]
    host_results = [r for r in results if r.get("role") == "host"]
    student_path = run_b / "survey_responses_student.jsonl"
    host_path = run_b / "survey_responses_host.jsonl"
    student_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in student_results),
        encoding="utf-8",
    )
    host_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in host_results),
        encoding="utf-8",
    )
    print(f"[ok] wrote {student_path} ({len(student_results)} student rows)")
    print(f"[ok] wrote {host_path} ({len(host_results)} host rows)")

    # 集計サマリ
    n_student_ok = sum(1 for r in student_results if r.get("responses"))
    n_host_ok = sum(1 for r in host_results if r.get("responses"))
    print(f"[summary] student responses: {n_student_ok}, host responses: {n_host_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
