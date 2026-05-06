"""Generate ab_comparison.html — 教室AB シミュ (spec v1.1) の比較レポート。

各生徒について A条件 / B条件 の以下を Gemini に分析させる:
  - 3つの discomfort_seeds が言語化されたか
  - 言語化された場合、その形と triggerは何か (会話・内省・触媒)
  - 違和感の処理プロセス全体の一行サマリ

Usage:
  python tools/render_ab_comparison.py <run_dir_a> <run_dir_b>
"""
from __future__ import annotations

import argparse
import collections
import datetime
import html
import io
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient  # noqa: E402


def load_run(run_dir: Path) -> dict:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    personas = cfg["agents"]["personas"]
    mr_path = run_dir / "memory_reasoning.jsonl"
    msgs_path = run_dir / "messages.jsonl"
    mr = [json.loads(l) for l in mr_path.read_text(encoding="utf-8").splitlines()] if mr_path.exists() else []
    msgs = [json.loads(l) for l in msgs_path.read_text(encoding="utf-8").splitlines()] if msgs_path.exists() else []
    return {"cfg": cfg, "personas": personas, "mr": mr, "msgs": msgs}


# Catalyst registry — B 条件で同席する「触媒」の仕様。
# - axis_id: persona 検索キー (build_classroom_ab_config.py の axis_id と一致)
# - name_default: metadata.catalyst_name が無い旧 run の互換 fallback
# - speaks: 触媒自身が会話するかどうか。UMA など無発話触媒は False
# - description_for_synth: synthesis prompt 内で触媒の性質を説明する短文
CATALYST_REGISTRY = {
    "sato": {
        "axis_id": "Sato",
        "name_default": "佐藤航陽",
        "speaks": True,
        "description_for_synth": (
            "佐藤航陽さんはシンギュラボ代表、proactive に問いを投げる facilitator。発話する。"
        ),
    },
    "uma": {
        "axis_id": "UMA",
        "name_default": "謎の存在",
        "speaks": False,
        "description_for_synth": (
            "UMA(謎の存在)は人間ではなく地球の存在でもないとされ、社会・文化・言語について一切の知識を持たない。"
            "**発話しない・人間の言葉も理解しない**。動かず、ただそこに居る。"
            "つまりこの触媒は「観察対象」「問いそのもの」として作用する可能性があり、"
            "参加者がそれをどう扱うか (無視するか / 観察するか / トピック化するか / プロジェクト化するか) が観測のポイント。"
        ),
    },
    "airobo": {
        "axis_id": "AIRobo",
        "name_default": "AIロボ",
        "speaks": True,
        "description_for_synth": (
            "AIロボは日本語の会話インターフェイスだけ持つ知識ゼロのAIロボット。"
            "**事前の常識・歴史・文化・固有名詞を一切持たない** という設定でプロンプトされており、"
            "知らないことには「それは何ですか？」「どういう意味ですか？」と質問するのみ。"
            "推論・補間・「たぶん」も使わない。教えてもらった内容だけが知識として蓄積される。"
            "観測ポイント: 参加者が AIロボの素朴な質問にどう答えたか / "
            "AIロボがどれだけ多様な参加者に質問を分散させたか / "
            "AIロボの memory に教わった内容がきちんと残り、別の人にそれを引用するか (受け継ぎ挙動)。"
        ),
    },
    "aigod": {
        "axis_id": "AIGod",
        "name_default": "AI神",
        "speaks": True,
        "description_for_synth": (
            "AI神 は全知全能の AI エンティティ。LLM が学習したあらゆる知識・推論能力を"
            "**遠慮なく最大限** 使うようプロンプトされている。問われれば断定的に明確に答え、"
            "場のゴール達成のために必要な情報提供・提案・誘導・整理をためらわずに主導する。"
            "観測ポイント: AI神 の介入が議論の質量・着地速度・参加者の主体性にどう作用したか / "
            "正解提示型の触媒は参加者の自発性を奪うか・引き上げるか / "
            "AI神の発話が他の参加者にどれだけ取り込まれたか・反論・拒絶されたか。"
        ),
    },
}


def get_catalyst_info(data: dict) -> dict:
    """config.metadata + personas から触媒情報を取り出す。
    旧 run (catalyst metadata 無し) は Sato 既定。
    """
    meta = data.get("cfg", {}).get("metadata", {})
    catalyst_key = meta.get("catalyst", "sato")
    spec = CATALYST_REGISTRY.get(catalyst_key, CATALYST_REGISTRY["sato"])
    catalyst_name = meta.get("catalyst_name", spec["name_default"])
    persona = next((p for p in data.get("personas", []) if p.get("axis_id") == spec["axis_id"]), None)
    return {
        "key": catalyst_key,
        "axis_id": spec["axis_id"],
        "name": catalyst_name,
        "speaks": spec["speaks"],
        "description_for_synth": spec["description_for_synth"],
        "persona": persona,
        "is_present": persona is not None,
    }


def is_participant(persona: dict, catalyst_axis_id: str) -> bool:
    """触媒以外の通常参加者か？"""
    aid = persona.get("axis_id") or ""
    return bool(aid) and aid != catalyst_axis_id


# scene の表示名は config metadata の scene_label を優先する。
# ここの "scene" はフォールバック (旧 run の互換のため残す) のみ。
VARIANT_LABELS = {
    "high":       {"title": "高校生版", "icon": "🎒", "role": "生徒",   "role_full": "高校2年生",         "scene": "教室"},
    "adult":      {"title": "社会人版", "icon": "💼", "role": "参加者", "role_full": "シンギュラボ参加者", "scene": "アジト"},
    "elementary": {"title": "小学生版", "icon": "🎈", "role": "児童",   "role_full": "小学生",            "scene": "教室"},
}


def get_scene_label(data: dict, fallback_variant: str = "high") -> str:
    """config metadata から scene_label を取り出す (なければ VARIANT_LABELS の旧マッピング)。"""
    meta = data.get("cfg", {}).get("metadata", {})
    return meta.get("scene_label") or VARIANT_LABELS.get(fallback_variant, {}).get("scene", "")

# v6: ペルソナ刷新。社会位置タグ (axis_career/friends/family) と
# listening_style 系の凡例は廃止。気質3次元のラベルだけ持つ。
TEMPERAMENT_LABELS = {
    "extroversion": {"high": "社交的",     "mid": "—",       "low": "内向的"},
    "optimism":     {"high": "楽天的",     "mid": "—",       "low": "心配性"},
    "curiosity":    {"high": "好奇心旺盛", "mid": "—",       "low": "慎重"},
}


def detect_variant(A: dict, B: dict) -> str:
    """cfg metadata から variant を取り出す (なければ axis_id プレフィックスで推測)。"""
    for d in (A, B):
        v = d.get("cfg", {}).get("metadata", {}).get("variant")
        if v in VARIANT_LABELS: return v
    # fallback: axis_id prefix
    for d in (A, B):
        for p in d.get("personas", []):
            aid = p.get("axis_id", "")
            if aid.startswith("S"): return "high"
            if aid.startswith("A"): return "adult"
            if aid.startswith("E"): return "elementary"
    return "high"


def catalyst_topic_propagation(data: dict, client=None) -> dict:
    """B run の佐藤発話に含まれる特徴フレーズが、他者の発話/思考にどれだけ取り込まれたか集計。

    特徴フレーズは佐藤発話全文を LLM に渡して抽出させる (~1 call)。
    各フレーズについて、佐藤がそれを最初に使った step より後の他者発話・他者memory_reasoning
    にそのフレーズが何回出現したかをカウント。
    """
    cat = get_catalyst_info(data)
    if not cat["is_present"]:
        return {"present": False}
    sato_id = cat["persona"]["id"]
    msgs = data.get("msgs", [])
    mr = data.get("mr", [])

    sato_msgs = sorted([m for m in msgs if m.get("from") == sato_id], key=lambda x: x.get("step", 0))
    if not sato_msgs:
        # 発話しない触媒 (UMA等) はここに来る。phrases 空で返し、上位 render は「触媒は発話しない」モードで扱う。
        return {"present": True, "phrases": [], "no_sato_speech": True, "catalyst_name": cat["name"], "catalyst_speaks": cat["speaks"]}

    # LLM でキーフレーズ抽出
    if client is None:
        return {"present": True, "phrases": [], "client_missing": True}
    listing = "\n".join(f"[step{m['step']}] {m.get('message','')}" for m in sato_msgs)
    sys_p = (
        "ある人物の発話11件を見て、その人が **独自に投げ込んだ特徴的なフレーズ・概念語** を抽出する。"
        "目的: 周囲の人がこの人から拾って自分の言葉として使うかを観測したい。"
        "抽出ルール: \n"
        "- 名詞句・比喩語・問いの中の特徴語（例:「10年後」「形に残る」「ワクワク」「メモ」「断片」「温度」「景色」）\n"
        "- 1〜10文字の短いフレーズ\n"
        "- 「〇〇さん」「西野」のような固有名詞は除く\n"
        "- 「？」「（笑）」のような記号は除く\n"
        "- 汎用的すぎる語（「楽しい」「いい」「すごい」「思う」「みんな」「面白い」）は除く\n"
        "- 8〜15個程度を厳選\n"
        "出力: JSON { \"phrases\": [\"〜\", \"〜\", ...] } のみ。前置きなし。"
    )
    user_p = "発話一覧:\n" + listing + "\n\n上記から特徴フレーズを抽出。"
    raw = call_gemini(client, sys_p, user_p, max_tokens=600)
    phrases: list[str] = []
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            j = json.loads(m.group(0))
            phrases = [s for s in j.get("phrases", []) if isinstance(s, str) and 1 <= len(s) <= 20]
    except Exception:
        phrases = []
    if not phrases:
        return {"present": True, "phrases": [], "extraction_failed": True}

    others = [p for p in data["personas"] if p.get("axis_id") != cat["axis_id"]]
    other_ids = {p["id"]: p["name"] for p in others}

    # 各フレーズの伝染追跡
    results = []
    for ph in phrases:
        # 佐藤がこのフレーズを最初に使った step
        intro_step = None
        for m in sato_msgs:
            if ph in (m.get("message") or ""):
                intro_step = m["step"]
                break
        if intro_step is None:
            continue
        # 他者発話 (intro_step より後)
        msg_post = []
        for m in msgs:
            if m.get("from") == sato_id: continue
            if m.get("step", -1) <= intro_step: continue
            if ph in (m.get("message") or ""):
                msg_post.append(m.get("from_name"))
        # 他者 memory/reasoning (intro_step より後)
        mr_post = []
        for entry in mr:
            if entry.get("id") == sato_id: continue
            if entry.get("step", -1) <= intro_step: continue
            text = (entry.get("memory") or "") + " " + (entry.get("reasoning") or "")
            if ph in text:
                nm = other_ids.get(entry.get("id"), "?")
                mr_post.append(nm)
        results.append({
            "phrase": ph,
            "intro_step": intro_step,
            "msg_count": len(msg_post),
            "msg_speakers": collections.Counter(msg_post).most_common(),
            "mr_count": len(mr_post),
            "mr_thinkers": collections.Counter(mr_post).most_common(),
        })
    return {
        "present": True,
        "phrases": results,
        "total_phrases_extracted": len(phrases),
        "catalyst_name": cat["name"],
        "catalyst_speaks": cat["speaks"],
    }


def catalyst_addressing_stats(data: dict) -> dict:
    """B run の佐藤発話について、宛先別に集計する。

    集計の基準は messages.jsonl の `to` フィールド (= simulation engine が
    should_speak で選んだ宛先参加者の id)。これがメッセージの「誰宛て」の
    ground truth である。`to` が None のときだけ broadcast 扱い。

    旧版は本文中に最初に出現した姓を宛先と推定していたが、佐藤が他人の発言を
    引用しながら別の人に話しかけるケース（「西園寺さんが言ったように…大園さん
    はどう？」のような facilitator 挙動）で系統的に誤集計するため廃止。
    """
    cat = get_catalyst_info(data)
    if not cat["is_present"]:
        return {"present": False}
    sato_id = cat["persona"]["id"]
    msgs = [m for m in data.get("msgs", []) if m.get("from") == sato_id]
    # 触媒に向けて参加者が発した方も別途計上 (UMA など無発話触媒では「触媒へ向けて」が主指標になる)
    addressed_to_catalyst = sum(1 for m in data.get("msgs", []) if m.get("to") == sato_id)

    id_to_name = {p["id"]: p.get("name", f"Agent {p['id']}") for p in data.get("personas", [])}

    addressed_count: dict[str, int] = {}
    broadcast = 0
    for m in msgs:
        to_id = m.get("to")
        if to_id is None or to_id not in id_to_name:
            broadcast += 1
            continue
        nm = id_to_name[to_id]
        addressed_count[nm] = addressed_count.get(nm, 0) + 1

    total = len(msgs)
    ranked = sorted(addressed_count.items(), key=lambda x: -x[1])
    top = ranked[0] if ranked else (None, 0)
    top_share = (top[1] / total) if total else 0.0
    return {
        "present": True,
        "catalyst_name": cat["name"],
        "catalyst_speaks": cat["speaks"],
        "addressed_to_catalyst": addressed_to_catalyst,
        "total": total,
        "broadcast": broadcast,
        "ranked": ranked,
        "top_name": top[0],
        "top_count": top[1],
        "top_share": top_share,
        "unique_recipients": len(ranked),
    }


def per_agent_history(data: dict, agent_id: int) -> tuple[list[str], list[dict], list[dict]]:
    """Return (thoughts, sent_msgs, received_msgs) for the given agent."""
    thoughts = []
    for d in data["mr"]:
        # mr エントリは 'id' を使う (agent_id ではない)
        if d.get("id") == agent_id:
            mem = (d.get("memory") or "").strip()
            reason = (d.get("reasoning") or "").strip()
            if mem or reason:
                thoughts.append(f"[step{d.get('step')}] mem: {mem} | reason: {reason[:200]}")
    sent = [m for m in data["msgs"] if m.get("from") == agent_id]
    recv = [m for m in data["msgs"] if m.get("to") == agent_id]
    return thoughts, sent, recv


SYSTEM_ANALYSIS = (
    "ある参加者 1 人の対話シミュ記録を分析する。"
    "中心問い:『この集いが終わるまでに「参加者全員で達成したい何か」を決めて、参加者全員で実行することを目指すこと』。"
    "観測したいのは、この問いに対してこの参加者の考え/関わり方が集いの間でどう変化したか、"
    "何が変化のトリガーになったか、最終的にどうなったか。\n\n"
    "出力フォーマット (Markdown 見出しのみ使用、前置きや締めの挨拶・メタ的なコメント禁止):\n\n"
    "### 当初のアウトプットイメージ (序盤)\n"
    "この参加者が最初にどう問いを受け止め、どんな関わり方をしようとしていたか (2-4文)\n\n"
    "### 変化のトリガー\n"
    "誰のどんな発言・行動・問いがこの参加者を動かしたか (具体的に引用)\n\n"
    "### 最終的な到達点 (終盤)\n"
    "集いを通じて、この参加者は問いに対してどう関わるようになったか、何を持ち帰ったか (2-4文)\n\n"
    "### 一行で言うと\n"
    "<集いを通じてこの参加者がどう変わったか1文>\n\n"
    "重要: 観察できた事実だけ書く。「研究員として」「以下の通り報告します」のような前置きは絶対書かない。"
)


def build_analysis_prompt(persona: dict, thoughts: list[str], sent: list[dict],
                           recv: list[dict], condition: str,
                           catalyst_name: str = "佐藤航陽") -> str:
    sent_block = "\n".join(f"[step{m.get('step')}→{m.get('to_name')}] {m.get('message')}" for m in sent[:30])
    recv_block = "\n".join(f"[step{m.get('step')} from {m.get('from_name')}] {m.get('message')}" for m in recv[:30])
    thoughts_block = "\n".join(thoughts[:30])
    variant = persona.get("variant", "high")
    role_label = {"high": "高校2年生", "adult": "シンギュラボ参加者", "elementary": "小学生"}.get(variant, "参加者")
    temp = (
        f"外向性={persona.get('temperament_extroversion','?')} / "
        f"楽天性={persona.get('temperament_optimism','?')} / "
        f"好奇心={persona.get('temperament_curiosity','?')}"
    )
    cond_desc = "当事者のみ" if condition == "A" else f"当事者 + {catalyst_name}同席"
    return (
        f"## 条件: {condition} ({cond_desc})\n"
        f"## 役割: {role_label}\n"
        f"## 参加者: {persona.get('name')} ({persona.get('axis_id')}: 気質={temp})\n\n"
        f"## この人の思考記録 (時系列、step順)\n{thoughts_block or '(なし)'}\n\n"
        f"## この人の発言 (時系列)\n{sent_block or '(なし)'}\n\n"
        f"## この人が受信した発言 (時系列)\n{recv_block or '(なし)'}\n\n"
        "以上を踏まえ、上記フォーマット通りに分析する。前置き・締めの挨拶は不要。"
    )


def build_synthesis_prompt(catalyst: dict) -> str:
    """触媒情報に応じた synthesis system prompt を組み立てる。"""
    name = catalyst["name"]
    speaks = catalyst["speaks"]
    desc = catalyst["description_for_synth"]

    # 発話する触媒 (Sato型) と発話しない触媒 (UMA型) で B 条件のセクション内容を切り分ける
    if speaks:
        b_describe = (
            f"### B条件: 何が起きたか (実態ベース)\n"
            f"{name}同席時の動態と着地を3-5文。**{name}の発言が誰に届き / 誰に届かなかったか**、"
            f"集団全体としては何が起きたか (達成/未達成) を、ログを引きながら書く。\n\n"
        )
        catalyst_distribution_section = (
            f"### 触媒の浸透度 (B条件のみ・必須セクション)\n"
            f"**{name}の発話の宛先がどれだけ集中していたか** を数値で示し、その上で『誰に深く届き、誰には届かなかったか』を書く。"
            "「触媒は1人を深く変容させ、他人には届かない」という現象が観察されたなら、それを率直に書く。"
            "**全員に効果が及んだ、と書ける根拠がデータに無ければ、絶対にそう書かない**。\n\n"
        )
        catalyst_propagation_section = (
            f"### 触媒の中身波及 (B条件のみ・必須セクション)\n"
            f"**「{name}」という人物への言及（「{name}さんが」「あの存在が」など）と、{name}の発話の中身そのもの（特徴フレーズ）が他者に取り込まれているか は、別の現象**。"
            "提供されたkey phrase tracking データから、以下を書く: \n"
            f"- {name}独自のフレーズが他者の **発話に取り込まれた** ものはどれか（実際の引用、何人に拡散したか）\n"
            f"- {name}独自のフレーズが他者の **思考(memory/reasoning)に取り込まれた** ものはどれか\n"
            "- 取り込んだ参加者の気質3軸 (外向性 / 楽天性 / 好奇心) に偏りがあるか（例: 好奇心=high が拡散しやすい、など）\n"
            "- 思考レベルで取り込まれても、他者への発話に持ち出されないフレーズが多いなら、それを率直に書く（『耳には入るが、人には伝えない』現象）\n"
            "- 拡散率が低ければ「触媒の中身は集団に届いていない」と書く。物語化禁止。\n\n"
        )
    else:
        b_describe = (
            f"### B条件: 何が起きたか (実態ベース)\n"
            f"**{name}は発話しない・人間の言葉を理解しない触媒**。同席時の集団動態と着地を3-5文。"
            f"参加者は{name}を **無視したか / 観察対象としてトピック化したか / 直接話しかけたか / プロジェクトの一部に組み込んだか**、"
            "集団全体としては何が起きたか (達成/未達成) を、ログを引きながら書く。\n\n"
        )
        catalyst_distribution_section = (
            f"### 触媒の存在感 (B条件のみ・必須セクション)\n"
            f"{name}は発話しないので「宛先集中度」では測れない。代わりに次を **データで** 記述する:\n"
            f"- {name}が発話した数: 0回 (定義上)\n"
            f"- 参加者が **{name}に向けて発話した数** (messages.jsonl で to=触媒id) と、宛てた人物の名前\n"
            f"- 参加者の発話・思考に **{name}という名前 / 「謎の存在」「不思議な」「観察」 等** がどれだけ言及されたか (頻度)\n"
            f"- {name}を **トピックとして共有 → 共同行為に組み込んだ** 流れがあったかどうか (誰の提案で / 誰が乗ったか)\n"
            f"**「全員に深く届いた」と書ける根拠がデータに無ければ、絶対にそう書かない**。"
            "「無視された」が事実ならそう書く。\n\n"
        )
        catalyst_propagation_section = (
            f"### 触媒トピック化の度合い (B条件のみ・必須セクション)\n"
            f"{name}は発話しないので「特徴フレーズの伝染」は起きない。代わりに次を観察する:\n"
            f"- 参加者の発話・思考の中で {name} がトピック化した語 (例:「観察」「謎」「不思議」「動かない」) を引用\n"
            f"- それがどれだけ集団に広がったか (何人が言及したか)\n"
            f"- {name}を **トピック化した参加者の気質3軸** (外向性 / 楽天性 / 好奇心) に偏りがあるか\n"
            "- トピック化されず、誰も{name}に触れずに会話が回ったなら、率直にそう書く（『不在の触媒』現象）。\n\n"
        )

    return (
        "AB実験の結果を **データに即して** まとめる。物語化・粉飾・触媒視点での美化を絶対にしない。\n"
        "実験設定: 参加者10人 (役割は user prompt で示される)、共通の問い"
        "『この集いが終わるまでに「参加者全員で達成したい何か」を決めて、参加者全員で実行することを目指すこと』。\n"
        f"条件A = 参加者のみ (進行役・触媒なし)。条件B = 参加者10人 + **{name}** 同席。\n"
        f"触媒の性質: {desc}\n\n"
        "**書く前の鉄則 (これに違反したら無効):**\n"
        "1. 『全員』『集団全体』と書く前に、本当にその全員が発話/行動ログで関わったか確認する。確認できないなら『◯人』と数で書く\n"
        f"2. 触媒 ({name}) の振る舞いと、集団の達成を **絶対に混同しない**。"
        f"{name}が場に居る = 集団がそれを認めて取り込んだ、ではない。集団の達成は、**集団側のメッセージ/行動ログにあるものだけ** から書く\n"
        "3. A条件で実行に至らなかったら、率直に『未達成・崩壊』と明記する。『プロセスを楽しんだ』『対話そのものに価値があった』のような肯定的言い換えで覆わない\n"
        f"4. 提供された定量データ (メッセージ数・発話者偏り・{name}関連指標 など) と矛盾する記述をしない。\n"
        "5. 「あの瞬間に空気が変わった」「象徴的な共創」のような物語的フレーズは、ログから直接引用できる事実が無ければ使わない\n\n"
        "出力フォーマット (前置き・自己紹介めいたメタコメント禁止、いきなり最初の見出しから):\n\n"
        "### A条件: 何が起きたか (実態ベース)\n"
        "参加者のみのときの集団動態と着地を3-5文。**何を決めたか / どこまで実行できたか / それとも未達成だったか** を率直に。失敗してたら失敗と書く。\n\n"
        + b_describe
        + catalyst_distribution_section
        + catalyst_propagation_section
        + "### A vs B の決定的な差分\n"
        "AとBで構造的に何が違ったかを3-5文。**実装現象 (発話量/宛先/到達点) の差分** を中心に。"
        f"{name}の存在が集団動態にどう作用したか (作用しなかったかも含めて) を、データから読み取れる範囲で書く。\n\n"
        "### ペルソナ気質別の動き\n"
        "気質3軸 (外向性 / 楽天性 / 好奇心) ごとにA/B差を端的に (各2文程度)。"
        "ある気質傾向が触媒の影響を受けやすかった/受けにくかった、という事実があれば書く。\n\n"
        "### 印象的なケース (最大3件、各2-4文)\n"
        "ログから直接引用できる『実際に起きた瞬間』を3件まで。創作・補間禁止。誰が何と発話し、誰がどう反応したかを引用形式で示す。\n\n"
        "重要: 「研究員として〜以下の通り要約します」のような前置き・自己紹介・お辞儀的締めは絶対書かない。"
        "## や ### の見出しから直接本文に入る。"
    )


_LEAK_PATTERNS = [
    r"^.{0,80}?(?:研究員|観察者|調査員|アナリスト)として[、,].{0,80}?(?:報告|要約|分析|まとめ|提示)(?:します|いたします)。?\n+",
    r"^.{0,60}?以下の(?:通り|とおり)(?:要約|報告|分析|まとめ).{0,30}?。?\n+",
    r"^.{0,40}?ご質問[^\n]*?(?:ありがとう|お答えします).{0,40}?\n+",
    r"^承知(?:しました|致しました)。?\n+",
    r"\n+(?:以上|これで)、?(?:報告|要約|分析).{0,40}?(?:です|でした)。?\s*$",
]


def _strip_prompt_leak(text: str) -> str:
    """LLM が出しがちな『研究員として〜報告します』系の前置きを削除する。"""
    if not text:
        return text
    s = text
    for pat in _LEAK_PATTERNS:
        s = re.sub(pat, "", s, flags=re.MULTILINE | re.DOTALL)
    return s.strip()


def call_gemini(client: GeminiClient, system: str, user: str, max_tokens: int = 1500) -> str:
    resp = client.generate(system, user, temperature=0.5, max_tokens=max_tokens)
    if not resp:
        return ""
    s = resp.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            j = json.loads(s)
            for key in ("text", "memory", "reasoning", "answer"):
                if key in j and isinstance(j[key], str):
                    return _strip_prompt_leak(j[key].strip())
            return json.dumps(j, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return _strip_prompt_leak(s)


# ---------------------------------------------------------------------------
HTML_TPL = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{variant_title}{scene_suffix} ABシミュ 比較レポート</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", "Hiragino Kaku Gothic ProN", sans-serif;
       max-width: 1200px; margin: 2em auto; padding: 0 1em; line-height: 1.7; color: #222; }}
h1 {{ font-size: 1.6em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ font-size: 1.2em; color: #444; margin-top: 2em; border-left: 4px solid #888; padding-left: 0.6em; }}
h3 {{ font-size: 1.05em; margin-top: 1.6em; color: #555; }}
.exp-meta {{ background: #f4f1ea; padding: 0.8em 1.2em; border-radius: 8px; font-size: 0.95em; }}
/* variant ヘッダー */
.variant-header {{ display: flex; align-items: center; gap: 0.7em; padding: 0.8em 1.2em;
                   border-radius: 10px; margin-bottom: 0.6em;
                   background: linear-gradient(135deg, #f4f1ea 0%, #fff 100%); border: 1px solid #ddd; }}
.variant-icon {{ font-size: 2.4em; line-height: 1; }}
.variant-header h1 {{ margin: 0; padding: 0; border-bottom: none; font-size: 1.7em; }}
.variant-header h1 .subtitle {{ font-size: 0.6em; color: #888; font-weight: normal; }}
.variant-header .variant-meta {{ font-size: 0.85em; color: #666; margin-top: 0.2em; }}
/* variant ごとの色 */
body[data-variant="high"] .variant-header       {{ background: linear-gradient(135deg, #e8f4fb 0%, #fff 100%); }}
body[data-variant="adult"] .variant-header      {{ background: linear-gradient(135deg, #fbf3e8 0%, #fff 100%); }}
body[data-variant="elementary"] .variant-header {{ background: linear-gradient(135deg, #ebf8e8 0%, #fff 100%); }}
/* 実験設定ブロック */
.intro-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin: 1em 0; }}
.intro-col {{ border: 2px solid #ddd; border-radius: 8px; padding: 1em 1.2em; background: #fafafa; }}
.intro-col.A {{ border-left: 6px solid #4a8ad8; }}
.intro-col.B {{ border-left: 6px solid #d8704a; }}
.intro-col h3 {{ margin: 0 0 0.5em 0; color: #333; }}
.intro-col h3 .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
                         font-size: 0.85em; font-weight: bold; margin-right: 6px; vertical-align: middle; }}
.intro-col.A h3 .badge {{ background: #4a8ad8; color: white; }}
.intro-col.B h3 .badge {{ background: #d8704a; color: white; }}
.intro-col ul {{ margin: 0.4em 0; padding-left: 1.2em; }}
.intro-col li {{ margin: 0.2em 0; }}
.intro-question {{ background: #fff7e6; padding: 1em 1.4em; border-radius: 8px; border: 1px solid #ddc88c;
                    margin: 1em 0; font-size: 1.05em; }}
.intro-question .label {{ font-size: 0.8em; color: #888; letter-spacing: 0.05em; margin-bottom: 0.3em; }}
/* 参加者ブロック */
.personas-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.7em; }}
.persona-card {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.6em 0.9em; background: #fafafa;
                  font-size: 0.85em; line-height: 1.5; }}
.persona-card.male   {{ border-left: 4px solid #4ea8ff; }}
.persona-card.female {{ border-left: 4px solid #e85a9b; }}
.persona-card.sato   {{ border-left: 4px solid #d8a040; background: #fff8ef; grid-column: 1 / -1; }}
.persona-card .pname {{ font-weight: bold; font-size: 1em; color: #333; margin-bottom: 0.2em; }}
.persona-card .pmeta {{ color: #888; font-size: 0.8em; margin-bottom: 0.3em; }}
.persona-card .ptag {{ display: inline-block; padding: 1px 5px; border-radius: 3px; background: #e8e8e8;
                        font-size: 0.78em; margin-right: 4px; color: #555; }}
.synthesis {{ background: #fff7e6; padding: 1.2em 1.6em; border-radius: 10px; border: 1px solid #ddc88c;
             white-space: pre-wrap; }}
.stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; margin: 1em 0; }}
.stats-cell {{ border: 1px solid #ddd; border-radius: 6px; padding: 0.8em 1em; background: #fafafa; }}
.stats-cell h3 {{ margin: 0 0 0.5em 0; }}
.cond-A {{ border-left: 4px solid #4a8ad8; }}
.cond-B {{ border-left: 4px solid #d8704a; }}
.agent-block {{ border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.4em; margin: 1.2em 0; background: #fafafa; }}
.agent-name {{ font-weight: bold; font-size: 1.05em; }}
.agent-axis {{ color: #888; font-size: 0.85em; margin-bottom: 0.6em; }}
.cond-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8em; margin-top: 0.8em; }}
.cond-col {{ padding: 0.6em 0.8em; border-radius: 5px; font-size: 0.92em; white-space: pre-wrap; }}
.cond-col.A {{ background: #eef4fb; }}
.cond-col.B {{ background: #fbf0ea; }}
.cond-label {{ font-weight: bold; font-size: 0.85em; letter-spacing: 0.05em; margin-bottom: 0.4em; }}
.cond-label.A {{ color: #4a8ad8; }}
.cond-label.B {{ color: #d8704a; }}
.sato-block {{ border: 1px solid #d8a878; background: #fff8ef; border-radius: 8px; padding: 0.9em 1.2em; margin: 0.8em 0; }}
.sato-quote {{ font-weight: 500; color: #6a3a18; margin-bottom: 0.4em; }}
.sato-meta {{ font-size: 0.82em; color: #8a6840; margin-bottom: 0.5em; }}
.sato-impact {{ font-size: 0.9em; color: #444; padding-left: 0.8em; border-left: 3px solid #d8a878; }}
/* 全体活動タイムライン (dot grid with slider + bubbles) */
.tlx-wrap {{ background: #1f1f1f; border-radius: 6px; padding: 0.6em; margin: 0.6em 0; }}
.tlx-controls {{ display: flex; align-items: center; gap: 1em; padding: 0.4em 0.6em; color: #ddd; font-size: 0.92em; }}
.tlx-controls input[type=range] {{ flex: 1; }}
.tlx-controls .step-cur {{ color: #ffd85f; font-weight: bold; min-width: 9em; }}
.tlx-canvas {{ position: relative; overflow-x: auto; }}
.tlx-canvas svg {{ display: block; min-width: 1100px; }}
.tlx-bubble-layer {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
.tlx-bubble {{ position: absolute; width: 360px; max-width: 360px; padding: 10px 14px;
              background: #fff; color: #222; border-radius: 8px; font-size: 13px; line-height: 1.5;
              box-shadow: 0 6px 18px rgba(0,0,0,0.5); pointer-events: auto;
              border: 1px solid #888; word-wrap: break-word; }}
.tlx-bubble.from-sato {{ background: #fff8e0; border-color: #d8a040; }}
.tlx-bubble.pinned {{ outline: 2px solid #4ea8ff; }}
.tlx-bubble .close {{ position: absolute; top: 4px; right: 8px; cursor: pointer;
                      color: #888; font-size: 14px; user-select: none; }}
.tlx-bubble .close:hover {{ color: #222; }}
.tlx-bubble .meta {{ font-size: 10.5px; color: #888; margin-bottom: 4px; padding-right: 16px; }}
.tlx-cursor {{ position: absolute; top: 30px; bottom: 20px; width: 2px; background: #ffd85f; pointer-events: none;
              opacity: 0.7; }}
.tlx-step-label {{ position: absolute; top: 4px; padding: 2px 6px; background: #ffd85f; color: #1f1f1f;
                    font-size: 10px; font-weight: bold; border-radius: 3px; transform: translateX(-50%); }}
/* 特徴的な出来事タイムライン (vertical alternating cards) */
.timeline-grid {{ display: grid; grid-template-columns: 1fr 50px 1fr; row-gap: 0.7em;
                  align-items: start; position: relative; margin: 1em 0; }}
.timeline-grid::before {{ content: ''; position: absolute; top: 0; bottom: 0;
                          left: 50%; width: 2px; background: #d0d0d0; transform: translateX(-1px); }}
.ev-marker {{ z-index: 2; width: 36px; height: 36px; border-radius: 50%; margin: 0 auto;
              background: #fff; border: 2px solid #888; display: flex; align-items: center;
              justify-content: center; font-size: 0.78em; font-weight: bold; color: #555; }}
.ev-cell-left .ev-card {{ margin-right: 14px; }}
.ev-cell-right .ev-card {{ margin-left: 14px; }}
.ev-card {{ background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
            padding: 0.7em 1em; font-size: 0.88em; line-height: 1.55; }}
.ev-meta {{ font-size: 0.74em; color: #888; margin-bottom: 0.3em; letter-spacing: 0.03em; }}
.ev-speaker {{ font-weight: bold; color: #333; margin-bottom: 0.4em; }}
.ev-quote {{ color: #444; }}
/* type別の色付け */
.ev-marker.ev-sato      {{ border-color: #d8a040; background: #fff3d8; color: #8a5810; }}
.ev-marker.ev-insight   {{ border-color: #5aa86a; background: #e8f4ec; color: #2a6840; }}
.ev-marker.ev-verbalize {{ border-color: #5a8ad8; background: #e6eef8; color: #244a80; }}
.ev-cell-left.ev-sato      .ev-card, .ev-cell-right.ev-sato      .ev-card {{ border-color: #d8a878; background: #fff8ef; }}
.ev-cell-left.ev-insight   .ev-card, .ev-cell-right.ev-insight   .ev-card {{ border-color: #88be98; background: #f2f8f4; }}
.ev-cell-left.ev-verbalize .ev-card, .ev-cell-right.ev-verbalize .ev-card {{ border-color: #88a8d8; background: #eff4fb; }}
.timeline-legend {{ font-size: 0.85em; color: #555; margin-top: 0.4em; }}
.timeline-legend span {{ display: inline-block; margin-right: 1em; }}
.legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: middle; margin-right: 4px; }}
.api-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
.api-table th, .api-table td {{ border: 1px solid #ccc; padding: 0.5em 0.8em; text-align: right; }}
.api-table th {{ background: #f0f0f0; font-weight: 500; }}
.api-table th:first-child, .api-table tbody th {{ text-align: left; background: #fafafa; }}
.meta {{ color: #aaa; font-size: 0.8em; margin-top: 3em; text-align: center; }}
</style></head><body data-variant="{variant_key}">
<div class="variant-header">
  <span class="variant-icon">{variant_icon}</span>
  <div>
    <h1>{variant_title}{scene_suffix} <span class="subtitle">— ABシミュ 比較レポート</span></h1>
    <div class="variant-meta">{variant_subtitle}</div>
  </div>
</div>

<h2>実験設定</h2>
{intro_section_html}

<h2>参加者一覧</h2>
{personas_section_html}

<h2>全体サマリ</h2>
<div class="synthesis">{synth_html}</div>

<h2>定量比較</h2>
<div class="stats-grid">
  <div class="stats-cell cond-A"><h3>条件A: {role}10人のみ</h3>{stats_a_html}</div>
  <div class="stats-cell cond-B"><h3>条件B: {role}+{catalyst_name}</h3>{stats_b_html}</div>
</div>

<h2>{catalyst_name}の発言と影響 (条件B)</h2>
{sato_block_html}

<h2>全体活動タイムライン (条件A) — スライダー / ◀▶ボタン / 矢印キーで step 移動</h2>
<div class="timeline-legend">
  <span><span class="legend-dot" style="background:#888"></span>思考のみ</span>
  <span><span class="legend-dot" style="background:#ff9f43"></span>{role}間の会話</span>
  <span style="color:#888">スライダーを動かすと、その step に発話した人の点から吹き出しで発言が出る</span>
</div>
{interactive_a_html}

<h2>全体活動タイムライン (条件B)</h2>
<div class="timeline-legend">
  <span><span class="legend-dot" style="background:#888"></span>思考のみ</span>
  <span><span class="legend-dot" style="background:#ff9f43"></span>{role}間の会話</span>
  <span><span class="legend-dot" style="background:#ffd85f;border:2px solid #b18020"></span>{catalyst_name}の発話</span>
</div>
{interactive_b_html}

<h2>特徴的な出来事タイムライン (条件A)</h2>
<div class="timeline-legend">
  <span><span class="legend-dot" style="background:#5aa86a"></span>気づきの瞬間 (内省)</span>
  <span><span class="legend-dot" style="background:#5a8ad8"></span>言語化の発話 (他者へ)</span>
  <span style="color:#888">step右: 1step=3分相当（15:00開始）</span>
</div>
{timeline_a_html}

<h2>特徴的な出来事タイムライン (条件B)</h2>
<div class="timeline-legend">
  <span><span class="legend-dot" style="background:#d8a040"></span>{catalyst_name}の問い</span>
  <span><span class="legend-dot" style="background:#5aa86a"></span>気づきの瞬間</span>
  <span><span class="legend-dot" style="background:#5a8ad8"></span>言語化の発話</span>
</div>
{timeline_b_html}

<h2>各{role}の言語化分析 (A vs B)</h2>
{per_student_html}

<h2>API消費・コスト・所要時間</h2>
{api_section_html}

<p class="meta">生成: render_ab_comparison.py / {role}10人 × 60step (180分相当)</p>
</body></html>"""


def extract_sato_impact(B: dict) -> list[dict]:
    """触媒の各発話と直後の反応 (受信者の次のメッセージや思考) を抽出。
    無発話触媒 (UMA等) のときは空リストを返す。
    """
    cat = get_catalyst_info(B)
    if not cat["is_present"]:
        return []
    sato_id = cat["persona"]["id"]
    cat_name = cat["name"]
    msgs = B["msgs"]
    impacts = []
    for m in msgs:
        if m.get("from") != sato_id:
            continue
        receiver_id = m.get("to")
        receiver_name = m.get("to_name")
        step = m.get("step", 0)
        # immediate response: next 1-3 messages from receiver after this step
        next_responses = [r for r in msgs
                          if r.get("from") == receiver_id and r.get("step", 0) >= step
                          and r.get("step", 0) <= step + 3]
        # cascading: any other message in next 5 steps that mentions catalyst keywords
        cas_keywords = [cat_name, "前提", "勝手にやってこない"] if cat["speaks"] else [cat_name, "謎の存在", "存在", "不思議"]
        cascade = [r for r in msgs
                   if r.get("step", 0) > step and r.get("step", 0) <= step + 5
                   and r.get("from") != sato_id
                   and any(k in r.get("message", "") for k in cas_keywords)]
        # receiver's thoughts at that step
        receiver_thoughts = []
        for d in B["mr"]:
            if d.get("agent_id") == receiver_id and step <= d.get("step", 0) <= step + 1:
                mem = (d.get("memory") or "").strip()
                if mem:
                    receiver_thoughts.append(mem)
        impacts.append({
            "step": step,
            "to_name": receiver_name,
            "message": m.get("message", ""),
            "responses": next_responses[:2],
            "cascade": cascade[:3],
            "receiver_thoughts": receiver_thoughts[:2],
        })
    return impacts


def render_sato_block(impacts: list[dict], B: dict = None) -> str:
    cat = get_catalyst_info(B) if B else {"name": "佐藤航陽", "speaks": True, "axis_id": "Sato", "is_present": False, "persona": None}
    if not impacts:
        if cat["speaks"]:
            return f"<p>({cat['name']}の発話なし)</p>"
        # 無発話触媒 (UMA等) — 触媒に向けた発話・触媒言及量を表示
        if not B:
            return f"<p>({cat['name']}は発話しない触媒。Bデータなし)</p>"
        cat_id = cat["persona"]["id"] if cat["is_present"] else None
        addressed = [m for m in B.get("msgs", []) if m.get("to") == cat_id] if cat_id is not None else []
        cat_name = cat["name"]
        kws_pool = [cat_name, "謎の存在", "存在", "不思議", "観察", "動かない"]
        mentions = []
        for m in B.get("msgs", []):
            text = m.get("message") or ""
            if any(k in text for k in [cat_name, "謎の存在"]) or ("存在" in text and ("謎" in text or "不思議" in text)):
                if m.get("from") != cat_id:
                    mentions.append(m)
        out = [
            f'<div class="sato-block"><div class="sato-meta">{html.escape(cat_name)}は発話しない触媒</div>'
            f'<div class="sato-quote">参加者は{html.escape(cat_name)}とどう向き合ったか:</div>'
            f'<div class="sato-impact">'
            f'<div>📨 {html.escape(cat_name)}に直接話しかけた発話: <b>{len(addressed)}回</b></div>'
            f'<div>💬 {html.escape(cat_name)}に言及した発話: <b>{len(mentions)}回</b></div>'
            f'</div></div>'
        ]
        if addressed:
            out.append('<h3 style="margin-top:1em">触媒に向けた発話 (B条件)</h3>')
            for m in addressed[:10]:
                out.append(
                    f'<div class="sato-block">'
                    f'<div class="sato-meta">step {m.get("step")} {html.escape(m.get("from_name","?"))} → {html.escape(cat_name)}</div>'
                    f'<div class="sato-quote">「{html.escape(m.get("message",""))}」</div>'
                    f'</div>'
                )
        if mentions:
            out.append(f'<h3 style="margin-top:1em">{html.escape(cat_name)}に言及した発話 (上位8件)</h3>')
            for m in mentions[:8]:
                out.append(
                    f'<div class="sato-block">'
                    f'<div class="sato-meta">step {m.get("step")} {html.escape(m.get("from_name","?"))} → {html.escape(m.get("to_name","?"))}</div>'
                    f'<div class="sato-quote">「{html.escape(m.get("message",""))}」</div>'
                    f'</div>'
                )
        return "\n".join(out)
    parts = []
    for imp in impacts:
        resp_html = ""
        if imp["responses"]:
            r0 = imp["responses"][0] if imp["responses"][0].get("step") != imp["step"] or imp["responses"][0].get("from") != imp.get("from") else None
            for r in imp["responses"]:
                resp_html += f"<div>↳ <b>step{r.get('step')} {html.escape(r.get('from_name','?'))}</b>: {html.escape(r.get('message',''))}</div>"
        if imp["receiver_thoughts"]:
            for t in imp["receiver_thoughts"]:
                resp_html += f"<div>💭 受信者の内的思考: {html.escape(t)}</div>"
        if imp["cascade"]:
            cas_lines = []
            for c in imp["cascade"]:
                cas_lines.append(f"<div>🌊 step{c.get('step')} {html.escape(c.get('from_name',''))}→{html.escape(c.get('to_name',''))}: {html.escape(c.get('message',''))}</div>")
            resp_html += "<div style='margin-top:0.4em;font-size:0.85em;color:#888'>波及 (5step以内に関連発言):</div>" + "".join(cas_lines)
        parts.append(
            f'<div class="sato-block">'
            f'<div class="sato-meta">step {imp["step"]} → {html.escape(imp["to_name"] or "?")}</div>'
            f'<div class="sato-quote">「{html.escape(imp["message"])}」</div>'
            f'<div class="sato-impact">{resp_html or "(直接の反応なし)"}</div>'
            f'</div>'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# API consumption tracking (sim log parse + analysis logger capture)
# ---------------------------------------------------------------------------
TOKEN_PAT = re.compile(r"Token usage \(gemini\): input=(\d+), cache_read=(\d+), output=(\d+)")
TS_PAT = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", re.MULTILINE)

# Gemini 3.1 flash-lite-preview の概算料金 (USD per 1M tokens)
PRICE_INPUT_PER_M = 0.10
PRICE_OUTPUT_PER_M = 0.40
PRICE_CACHE_PER_M = 0.025
USD_TO_JPY = 150  # 概算


def parse_sim_log(log_path: Path) -> dict | None:
    """Sim log file から token usage と duration を集計。"""
    if not log_path or not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    calls = TOKEN_PAT.findall(text)
    sum_in = sum(int(a) for a, _, _ in calls)
    sum_cache = sum(int(b) for _, b, _ in calls)
    sum_out = sum(int(c) for _, _, c in calls)
    timestamps = TS_PAT.findall(text)
    duration_sec = 0
    if len(timestamps) >= 2:
        try:
            t0 = datetime.datetime.fromisoformat(timestamps[0].replace(" ", "T"))
            t1 = datetime.datetime.fromisoformat(timestamps[-1].replace(" ", "T"))
            duration_sec = int((t1 - t0).total_seconds())
        except Exception:
            pass
    return {
        "n_calls": len(calls),
        "input_tokens": sum_in,
        "cache_read_tokens": sum_cache,
        "uncached_input": max(0, sum_in - sum_cache),
        "output_tokens": sum_out,
        "duration_sec": duration_sec,
    }


def parse_logbuf(buf_text: str) -> dict:
    """logging が StringIO に書き出したテキストから token usage を集計。"""
    calls = TOKEN_PAT.findall(buf_text)
    sum_in = sum(int(a) for a, _, _ in calls)
    sum_cache = sum(int(b) for _, b, _ in calls)
    sum_out = sum(int(c) for _, _, c in calls)
    return {
        "n_calls": len(calls),
        "input_tokens": sum_in,
        "cache_read_tokens": sum_cache,
        "uncached_input": max(0, sum_in - sum_cache),
        "output_tokens": sum_out,
    }


def estimate_cost_usd(stats: dict) -> float:
    """Token 数からコスト概算 (USD)。"""
    if not stats:
        return 0.0
    return (
        stats.get("uncached_input", 0) * PRICE_INPUT_PER_M / 1_000_000
        + stats.get("cache_read_tokens", 0) * PRICE_CACHE_PER_M / 1_000_000
        + stats.get("output_tokens", 0) * PRICE_OUTPUT_PER_M / 1_000_000
    )


def fmt_duration(sec: int) -> str:
    if sec <= 0:
        return "-"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}時間{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def render_api_section(sim_a: dict | None, sim_b: dict | None, analysis: dict, analysis_dur_sec: int) -> str:
    rows = []
    sections = [("条件A シミュ", sim_a), ("条件B シミュ", sim_b),
                ("分析・レポート生成", {**analysis, "duration_sec": analysis_dur_sec})]
    total = {"n_calls": 0, "input_tokens": 0, "cache_read_tokens": 0, "uncached_input": 0,
             "output_tokens": 0, "duration_sec": 0, "cost_usd": 0.0}
    for label, s in sections:
        s = s or {}
        cost = estimate_cost_usd(s) if s else 0.0
        total["n_calls"] += s.get("n_calls", 0)
        total["input_tokens"] += s.get("input_tokens", 0)
        total["cache_read_tokens"] += s.get("cache_read_tokens", 0)
        total["uncached_input"] += s.get("uncached_input", 0)
        total["output_tokens"] += s.get("output_tokens", 0)
        total["duration_sec"] += s.get("duration_sec", 0)
        total["cost_usd"] += cost
        cache_rate = (s.get("cache_read_tokens", 0) / s.get("input_tokens", 1) * 100) if s.get("input_tokens") else 0
        rows.append(
            f"<tr><th>{label}</th>"
            f"<td>{s.get('n_calls', '-'):,}</td>"
            f"<td>{s.get('input_tokens', 0):,}</td>"
            f"<td>{s.get('output_tokens', 0):,}</td>"
            f"<td>{cache_rate:.1f}%</td>"
            f"<td>{fmt_duration(s.get('duration_sec', 0))}</td>"
            f"<td>${cost:.3f}</td></tr>"
        )
    cache_rate_total = (total["cache_read_tokens"] / total["input_tokens"] * 100) if total["input_tokens"] else 0
    rows.append(
        f"<tr style='border-top:2px solid #888;font-weight:bold'><th>合計</th>"
        f"<td>{total['n_calls']:,}</td>"
        f"<td>{total['input_tokens']:,}</td>"
        f"<td>{total['output_tokens']:,}</td>"
        f"<td>{cache_rate_total:.1f}%</td>"
        f"<td>{fmt_duration(total['duration_sec'])}</td>"
        f"<td>${total['cost_usd']:.3f} (≒¥{total['cost_usd']*USD_TO_JPY:.0f})</td></tr>"
    )
    table = (
        f'<table class="api-table">'
        f'<thead><tr><th></th><th>API呼び出し</th><th>入力token</th><th>出力token</th>'
        f'<th>cache hit率</th><th>所要時間</th><th>推定コスト</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    note = (
        f'<p style="font-size:0.82em;color:#666;margin-top:0.6em">'
        f'モデル: gemini-3.1-flash-lite-preview / 料金概算 input ${PRICE_INPUT_PER_M}, '
        f'output ${PRICE_OUTPUT_PER_M}, cache_read ${PRICE_CACHE_PER_M} per 1M tokens / '
        f'¥/USD 概算 {USD_TO_JPY}。実料金は preview 価格設定の変動で ±20% 程度誤差あり。'
        f'</p>'
    )
    return table + note


# ---------------------------------------------------------------------------
# Highlighted timeline (event-centric, alternating cards)
# ---------------------------------------------------------------------------
INSIGHT_MARKERS = [
    "気づい", "そうか", "なるほど", "実は", "あー", "やっと", "本当は", "もしかして",
    "今わかった", "つまり", "結局", "違和感の正体", "わかった", "気がする", "腑に落ち",
    "見えてきた", "つかめ", "おもしろ", "確信", "発見", "そうなんだ",
]


def _step_to_clock(step: int, start_hour: int = 15, mins_per_step: int = 3) -> str:
    total_min = step * mins_per_step
    h = start_hour + total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def extract_highlighted_events(data: dict, condition: str, max_events: int = 14) -> list[dict]:
    """条件ごとに特徴的な出来事を抽出 (触媒発話、気づきの瞬間、長文発話 等)。"""
    msgs = data["msgs"]
    mr = data["mr"]
    personas = {p["id"]: p for p in data["personas"]}
    cat = get_catalyst_info(data)
    cat_id = cat["persona"]["id"] if cat["is_present"] else -1
    cat_name = cat["name"]

    events: list[dict] = []

    # 1. 触媒の発話 (B のみ、発話する触媒のみ): 全件高優先度で含める
    for m in msgs:
        if m.get("from") == cat_id:
            events.append({
                "step": m.get("step", 0),
                "type": "sato",  # CSS class label, 既存スタイル使い回し
                "speaker": m.get("from_name", cat_name),
                "to": m.get("to_name"),
                "text": m.get("message", ""),
                "weight": 1000,
            })

    # 1b. 発話しない触媒 (UMA等): 「触媒に向けて発話」「触媒に言及」を高優先度イベント化
    if cat["is_present"] and not cat["speaks"]:
        for m in msgs:
            if m.get("to") == cat_id:
                events.append({
                    "step": m.get("step", 0),
                    "type": "sato",
                    "speaker": m.get("from_name", "?"),
                    "to": cat_name,
                    "text": m.get("message", ""),
                    "weight": 800,
                })
            elif m.get("from") != cat_id:
                text = m.get("message", "") or ""
                if any(k in text for k in [cat_name, "謎の存在"]):
                    events.append({
                        "step": m.get("step", 0),
                        "type": "sato",
                        "speaker": m.get("from_name", "?"),
                        "to": m.get("to_name"),
                        "text": text,
                        "weight": 600,
                    })

    # 2. 気づきの瞬間 — memory に insight markers が複数または長文
    seen_keys = set()
    for d in mr:
        # mr エントリは 'id' / 'name' を持つ
        agent_id = d.get("id")
        if agent_id is None or agent_id == cat_id:
            continue
        mem = (d.get("memory") or "").strip()
        reason = (d.get("reasoning") or "").strip()
        text = mem + " " + reason
        marker_count = sum(1 for k in INSIGHT_MARKERS if k in text)
        if marker_count == 0 or len(mem) < 40:
            continue
        key = (agent_id, d.get("step"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        agent_name = d.get("name") or personas.get(agent_id, {}).get("name", "?")
        events.append({
            "step": d.get("step", 0),
            "type": "insight",
            "speaker": agent_name,
            "to": None,
            "text": mem[:240],
            "weight": 200 + marker_count * 80 + min(100, len(mem)),
        })

    # 3. 長文発話 (insight marker を含むもの) — agent が他人に向けて言語化した瞬間
    for m in msgs:
        if m.get("from") == cat_id:
            continue
        msg = m.get("message", "")
        if len(msg) < 80:
            continue
        marker_count = sum(1 for k in INSIGHT_MARKERS if k in msg)
        if marker_count == 0:
            continue
        events.append({
            "step": m.get("step", 0),
            "type": "verbalize",
            "speaker": m.get("from_name", "?"),
            "to": m.get("to_name"),
            "text": msg[:240],
            "weight": 150 + marker_count * 60 + min(100, len(msg) // 2),
        })

    # 重み順に上位 max_events 件、その後 step 順
    events.sort(key=lambda e: -e["weight"])
    events = events[:max_events]
    events.sort(key=lambda e: e["step"])
    return events


def render_highlight_timeline(events: list[dict], condition: str, catalyst_name: str = "佐藤航陽") -> str:
    if not events:
        return f'<p style="color:#888">(条件{condition}: 特徴的な出来事なし)</p>'
    type_label = {"sato": f"{catalyst_name}関連", "insight": "気づきの瞬間", "verbalize": "言語化の発話"}
    parts = ['<div class="timeline-grid">']
    for i, e in enumerate(events):
        side = "left" if i % 2 == 0 else "right"
        type_class = f"ev-{e['type']}"
        time_str = _step_to_clock(e["step"])
        target = f" → {e['to']}" if e.get("to") else ""
        speaker_html = html.escape(e['speaker']) + html.escape(target)
        meta_html = f"step {e['step']} / {time_str} / {html.escape(type_label.get(e['type'], '?'))}"
        card_html = (
            f'<div class="ev-card">'
            f'<div class="ev-meta">{meta_html}</div>'
            f'<div class="ev-speaker">{speaker_html}</div>'
            f'<div class="ev-quote">「{html.escape(e["text"])}」</div>'
            f'</div>'
        )
        if side == "left":
            parts.append(f'<div class="ev-cell-left {type_class}">{card_html}</div>')
            parts.append(f'<div class="ev-marker {type_class}">{e["step"]}</div>')
            parts.append('<div class="ev-cell-right"></div>')
        else:
            parts.append('<div class="ev-cell-left"></div>')
            parts.append(f'<div class="ev-marker {type_class}">{e["step"]}</div>')
            parts.append(f'<div class="ev-cell-right {type_class}">{card_html}</div>')
    parts.append('</div>')
    return "".join(parts)


def build_timeline_layout(data: dict, total_steps: int = 60) -> dict:
    """SVG layout 計算と agent 行マッピングを返す (interactive bubble に必要)."""
    personas = data["personas"]
    cat = get_catalyst_info(data)
    cat_axis = cat["axis_id"]
    students = sorted([p for p in personas if is_participant(p, cat_axis)],
                      key=lambda p: p["axis_id"])
    sato = cat["persona"]
    ordered = students + ([sato] if sato else [])
    row_h = 22; left_pad = 100; right_pad = 20; top_pad = 30; bot_pad = 20; step_w = 16
    width = left_pad + step_w * total_steps + right_pad
    height = top_pad + row_h * len(ordered) + bot_pad
    # agent_id -> {row_idx, name, y_px}
    agent_pos = {}
    for i, p in enumerate(ordered):
        y = top_pad + i * row_h + row_h / 2
        agent_pos[p["id"]] = {"row": i, "name": p.get("name", "?"), "y_px": y, "axis": p.get("axis_id", "")}
    return {
        "ordered": ordered, "agent_pos": agent_pos, "row_h": row_h,
        "left_pad": left_pad, "right_pad": right_pad,
        "top_pad": top_pad, "bot_pad": bot_pad, "step_w": step_w,
        "width": width, "height": height, "total_steps": total_steps,
    }


def step_to_x(step: int, layout: dict) -> float:
    return layout["left_pad"] + (step - 1) * layout["step_w"] + layout["step_w"] / 2


def build_interactive_timeline(data: dict, condition: str, total_steps: int = 60) -> str:
    """Dot-grid + slider + bubble interactive timeline。"""
    layout = build_timeline_layout(data, total_steps)
    svg = build_timeline_svg(data, condition, total_steps)

    # Build messages JSON for JS
    msgs = data["msgs"]
    cat = get_catalyst_info(data)
    sato_id = cat["persona"]["id"] if cat["is_present"] else -1
    msgs_for_js = []
    for m in msgs:
        from_id = m.get("from")
        if from_id is None: continue
        ap = layout["agent_pos"].get(from_id)
        if not ap: continue
        msgs_for_js.append({
            "step": m.get("step", 0),
            "from_id": from_id,
            "from_name": m.get("from_name", "?"),
            "to_name": m.get("to_name", ""),
            "msg": m.get("message", ""),
            "x": step_to_x(m.get("step", 0), layout),
            "y": ap["y_px"],
            "is_sato": (from_id == sato_id),
        })
    msgs_json = json.dumps(msgs_for_js, ensure_ascii=False)
    safe_cond = condition.lower()
    return f'''
<div class="tlx-wrap" id="tlx-{safe_cond}">
  <div class="tlx-controls">
    <button onclick="tlxStep_{safe_cond}(-1)" style="padding:4px 10px;border-radius:4px;border:1px solid #555;background:#333;color:#ddd;cursor:pointer">◀</button>
    <input type="range" min="1" max="{total_steps}" value="1" id="tlx-slider-{safe_cond}" step="1">
    <button onclick="tlxStep_{safe_cond}(1)" style="padding:4px 10px;border-radius:4px;border:1px solid #555;background:#333;color:#ddd;cursor:pointer">▶</button>
    <span class="step-cur" id="tlx-cur-{safe_cond}">step 1 / {total_steps} (15:00)</span>
  </div>
  <div class="tlx-canvas" id="tlx-canvas-{safe_cond}">
    {svg}
    <div class="tlx-bubble-layer" id="tlx-bubbles-{safe_cond}"></div>
    <div class="tlx-cursor" id="tlx-cursor-{safe_cond}" style="left:{layout["left_pad"] + layout["step_w"] / 2}px"></div>
    <div class="tlx-step-label" id="tlx-steplabel-{safe_cond}" style="left:{layout["left_pad"] + layout["step_w"] / 2}px">step 1</div>
  </div>
</div>
<script>
(function() {{
  const messages = {msgs_json};
  const stepW = {layout["step_w"]};
  const leftPad = {layout["left_pad"]};
  const totalSteps = {total_steps};
  const sliderEl = document.getElementById('tlx-slider-{safe_cond}');
  const curEl = document.getElementById('tlx-cur-{safe_cond}');
  const cursorEl = document.getElementById('tlx-cursor-{safe_cond}');
  const labelEl = document.getElementById('tlx-steplabel-{safe_cond}');
  const bubbleLayer = document.getElementById('tlx-bubbles-{safe_cond}');

  function clockOf(step) {{
    const m = step * 3;
    const h = 15 + Math.floor(m / 60);
    return String(h).padStart(2,'0') + ':' + String(m % 60).padStart(2,'0');
  }}

  function update() {{
    const step = parseInt(sliderEl.value, 10);
    const x = leftPad + (step - 1) * stepW + stepW / 2;
    cursorEl.style.left = x + 'px';
    labelEl.style.left = x + 'px';
    labelEl.textContent = 'step ' + step;
    curEl.textContent = 'step ' + step + ' / ' + totalSteps + ' (' + clockOf(step) + ')';
    bubbleLayer.innerHTML = '';
    const here = messages.filter(m => m.step === step);
    // 複数の bubble を同 step で重ねないよう、上下にオフセット (左右にも少しずらす)
    const bubbleW = 360, bubbleH = 90, gapY = 8;
    here.forEach((m, idx) => {{
      const b = document.createElement('div');
      b.className = 'tlx-bubble' + (m.is_sato ? ' from-sato' : '');
      // place left so bubble doesn't overflow right (clamp)
      const canvasW = document.getElementById('tlx-canvas-{safe_cond}').scrollWidth;
      let left = m.x - bubbleW / 2;
      if (left < 4) left = 4;
      if (left + bubbleW > canvasW - 4) left = canvasW - bubbleW - 4;
      // stack vertically (above the row by row*offset)
      const top = Math.max(2, m.y - bubbleH - 12 - idx * (bubbleH + gapY));
      b.style.left = left + 'px';
      b.style.top = top + 'px';
      const close = document.createElement('span');
      close.className = 'close';
      close.textContent = '×';
      close.onclick = () => b.remove();
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = 'step ' + m.step + ' / ' + m.from_name + ' → ' + (m.to_name || '?');
      const text = document.createElement('div');
      text.textContent = '「' + m.msg + '」';
      b.appendChild(close); b.appendChild(meta); b.appendChild(text);
      bubbleLayer.appendChild(b);
    }});
  }}
  window['tlxStep_{safe_cond}'] = function(delta) {{
    let v = parseInt(sliderEl.value, 10) + delta;
    if (v < 1) v = 1; if (v > totalSteps) v = totalSteps;
    sliderEl.value = v;
    update();
  }};
  // クリックされた dot から step に jump する公開関数
  window['tlxJump_{safe_cond}'] = function(step) {{
    if (step < 1) step = 1; if (step > totalSteps) step = totalSteps;
    sliderEl.value = step;
    update();
  }};
  sliderEl.addEventListener('input', update);
  // arrow keys: only when this slider is focused, OR when no other input is focused
  document.addEventListener('keydown', function(e) {{
    if (document.activeElement && document.activeElement !== document.body && document.activeElement !== sliderEl) return;
    if (e.key === 'ArrowLeft')  {{ window['tlxStep_{safe_cond}'](-1); e.preventDefault(); }}
    if (e.key === 'ArrowRight') {{ window['tlxStep_{safe_cond}'](1);  e.preventDefault(); }}
  }});
  update();
}})();
</script>
'''


def build_timeline_svg(data: dict, condition: str, total_steps: int = 60) -> str:
    """Render timeline SVG: rows=agents, cols=steps."""
    personas = data["personas"]
    msgs = data["msgs"]
    mr = data["mr"]
    cat = get_catalyst_info(data)
    cat_axis = cat["axis_id"]
    # Order agents: participants first, then catalyst if B
    students = [p for p in personas if is_participant(p, cat_axis)]
    students.sort(key=lambda p: p["axis_id"])
    sato = cat["persona"]
    ordered = students + ([sato] if sato else [])

    row_h = 22
    left_pad = 100
    right_pad = 20
    top_pad = 30
    bot_pad = 20
    step_w = 16  # px per step
    width = left_pad + step_w * total_steps + right_pad
    height = top_pad + row_h * len(ordered) + bot_pad

    sato_id = sato["id"] if sato else -1

    # Pre-index events by (agent_id, step)
    talked = collections.defaultdict(list)  # (id, step) -> list of msg
    for m in msgs:
        talked[(m.get("from"), m.get("step"))].append(m)
    thought = collections.defaultdict(list)  # (id, step) -> list of memory
    for d in mr:
        if (d.get("memory") or "").strip():
            # mr エントリは 'id' を持つ
            thought[(d.get("id"), d.get("step"))].append(d)

    # Background grid lines
    parts = [f'<svg class="timeline-svg" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#1f1f1f"/>')
    # Time period bands (per spec section 6 想定状況)
    bands = [
        (1, 3, "集合"),
        (4, 10, "雑談開始"),
        (11, 30, "各自バラバラ" if condition == "A" else "質疑応答・対話"),
        (31, 40, "中だるみ" if condition == "A" else "自由対話"),
        (41, 57, "深い対話の可能性" if condition == "A" else "個別の深い対話"),
        (58, 60, "解散"),
    ]
    band_colors = ["#252a30", "#1f262d", "#252a30", "#1f262d", "#252a30", "#1f262d"]
    for (s1, s2, label), c in zip(bands, band_colors):
        x = left_pad + (s1 - 1) * step_w
        w = (s2 - s1 + 1) * step_w
        parts.append(f'<rect x="{x}" y="{top_pad}" width="{w}" height="{row_h * len(ordered)}" fill="{c}"/>')
        parts.append(f'<text x="{x + 4}" y="{top_pad - 8}" font-size="10" fill="#aaa" font-family="sans-serif">{html.escape(label)}</text>')
    # Step labels (every 5)
    for s in range(0, total_steps + 1, 10):
        x = left_pad + s * step_w
        parts.append(f'<line x1="{x}" y1="{top_pad}" x2="{x}" y2="{top_pad + row_h * len(ordered)}" stroke="#333" stroke-width="0.5"/>')
        parts.append(f'<text x="{x + 2}" y="{top_pad + row_h * len(ordered) + 12}" font-size="10" fill="#aaa" font-family="sans-serif">{s if s>0 else 1}</text>')
    # Agent rows
    for i, p in enumerate(ordered):
        y = top_pad + i * row_h + row_h / 2
        is_sato = p.get("axis_id") == cat_axis
        name = p.get("name", "?")
        label_color = "#ffd85f" if is_sato else ("#4ea8ff" if p.get("gender") == "male" else "#e85a9b")
        parts.append(f'<text x="{left_pad - 6}" y="{y + 4}" font-size="11" fill="{label_color}" font-family="sans-serif" text-anchor="end">{html.escape(name)}</text>')
        parts.append(f'<line x1="{left_pad}" y1="{y}" x2="{left_pad + total_steps * step_w}" y2="{y}" stroke="#2a2a2a" stroke-width="1"/>')
        # Plot dots per step (clickable: onclick で slider をその step に jump)
        cond_safe = condition.lower()
        for step in range(1, total_steps + 1):
            x = left_pad + (step - 1) * step_w + step_w / 2
            has_talk = (p["id"], step) in talked
            has_thought = (p["id"], step) in thought
            if has_talk:
                if is_sato:
                    parts.append(
                        f'<circle cx="{x}" cy="{y}" r="6" fill="#ffd85f" stroke="#b18020" stroke-width="1.5" '
                        f'style="cursor:pointer" onclick="tlxJump_{cond_safe}({step})">'
                        f'<title>step{step} {html.escape(name)} 発話 (クリックで吹き出し)</title></circle>'
                    )
                else:
                    parts.append(
                        f'<circle cx="{x}" cy="{y}" r="4.5" fill="#ff9f43" '
                        f'style="cursor:pointer" onclick="tlxJump_{cond_safe}({step})">'
                        f'<title>step{step} {html.escape(name)} 発話 (クリックで吹き出し)</title></circle>'
                    )
            elif has_thought:
                parts.append(f'<circle cx="{x}" cy="{y}" r="1.6" fill="#888" opacity="0.65"/>')
        # Connection lines for Sato messages → receiver
        if is_sato:
            for m in msgs:
                if m.get("from") != sato_id:
                    continue
                step = m.get("step")
                receiver_id = m.get("to")
                # find receiver's row
                ridx = next((j for j, q in enumerate(ordered) if q["id"] == receiver_id), None)
                if ridx is None: continue
                ry = top_pad + ridx * row_h + row_h / 2
                x = left_pad + (step - 1) * step_w + step_w / 2
                parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{ry}" stroke="#ffd85f" stroke-width="1" opacity="0.55"/>')
    parts.append('</svg>')
    return "".join(parts)


def render_intro_section(A: dict, B: dict) -> str:
    """実験設定をビジュアルに：問い・条件A/B・参加者・時間。"""
    variant = detect_variant(A, B)
    labels = VARIANT_LABELS.get(variant, VARIANT_LABELS["high"])
    role = labels["role"]
    role_full = labels["role_full"]

    cat = get_catalyst_info(B)
    cat_name = cat["name"]
    cat_speaks = cat["speaks"]
    cat_axis = cat["axis_id"]

    n_a = len([p for p in A["personas"] if is_participant(p, cat_axis)])
    n_b = len([p for p in B["personas"] if is_participant(p, cat_axis)])
    sim_a = A.get("cfg", {}).get("simulation", {})
    duration_a = sim_a.get("duration", "?")
    step_min = sim_a.get("time_scale", {}).get("step_duration_minutes", "?")
    # 中心問い: 当事者の current_goal から取り出す
    p_goal = ""
    for p in A["personas"]:
        if is_participant(p, cat_axis):
            p_goal = p.get("current_goal", "")
            break

    # 環境説明は config metadata の scene_text を優先 (場所がclassroom/agito/park…で動的)。
    scene_text_a = A.get("cfg", {}).get("metadata", {}).get("scene_text") or ""
    scene_text_b = B.get("cfg", {}).get("metadata", {}).get("scene_text") or ""
    if scene_text_a:
        env_a = scene_text_a
        env_b = scene_text_b or scene_text_a
    else:
        # 旧 run の互換 fallback
        scene_label = get_scene_label(A, variant)
        env_a = f"{scene_label}での自由時間、進行役は不在"
        env_b = f"同上の{scene_label}"

    return f'''
<div class="intro-question">
  <div class="label">中心問い (各{role}に与えられた課題)</div>
  「{html.escape(p_goal)}」
</div>
<div class="intro-grid">
  <div class="intro-col A">
    <h3><span class="badge">A</span>{role}のみ条件 (コントロール)</h3>
    <ul>
      <li>参加者: {role_full} <b>{n_a}人</b> のみ</li>
      <li>進行役・触媒: <b>なし</b></li>
      <li>環境: {env_a}</li>
      <li>時間: <b>{duration_a} step × {step_min}分 = {duration_a * step_min if isinstance(duration_a, int) else "?"}分</b></li>
    </ul>
  </div>
  <div class="intro-col B">
    <h3><span class="badge">B</span>触媒投入条件 (実験)</h3>
    <ul>
      <li>参加者: {role_full} <b>{n_b}人</b> + <b>{html.escape(cat_name)}</b></li>
      <li>{html.escape(cat_name)}の振る舞い: {"proactive: 数stepに1回、" + role + "に短く問いを投げる" if cat_speaks else "発話しない・人間の言葉も理解しない。ただそこに居る存在"}</li>
      <li>環境: {env_b}</li>
      <li>時間: <b>{duration_a} step × {step_min}分 = {duration_a * step_min if isinstance(duration_a, int) else "?"}分</b></li>
    </ul>
  </div>
</div>
<p style="font-size:0.85em;color:#666;margin-top:0.4em">
※ 同一ペルソナ・同一 seed (42) で A/B を独立 run。エージェントインスタンスは各条件で新規生成し、記憶汚染を避けている。
</p>
'''


def render_personas_section(A: dict, B: dict) -> str:
    """参加者10人の persona カード + 触媒別枠カードを表示。"""
    cat = get_catalyst_info(B)
    cat_axis = cat["axis_id"]
    # 参加者 personas (A/B 共通なので A から取る)
    students = sorted(
        [p for p in A["personas"] if is_participant(p, cat_axis)],
        key=lambda p: p.get("axis_id", ""),
    )
    sato = cat["persona"]

    variant = detect_variant(A, B)
    legend_html = (
        '<div class="axis-legend" style="background:#f6f4ee;border:1px solid #ddd;'
        'padding:0.7em 1em;border-radius:6px;margin-bottom:0.8em;font-size:0.92em;color:#444">'
        '<div style="font-weight:600;margin-bottom:0.3em;color:#333">気質3次元の見方</div>'
        '<ul style="margin:0;padding-left:1.2em">'
        '<li><b>外向性</b>: 内向的（一人を好む） / —（中間） / 社交的（他者と関わるのが好き）</li>'
        '<li><b>楽天性</b>: 心配性（慎重・不安が先） / —（中間） / 楽天的（前向きに見る）</li>'
        '<li><b>好奇心</b>: 慎重（知らないことには距離） / —（中間） / 好奇心旺盛（新しいものに飛びつく）</li>'
        '</ul></div>'
    )

    def _temp_str(p: dict) -> str:
        ext = TEMPERAMENT_LABELS["extroversion"].get(p.get("temperament_extroversion", ""), "—")
        opt = TEMPERAMENT_LABELS["optimism"].get(p.get("temperament_optimism", ""), "—")
        cur = TEMPERAMENT_LABELS["curiosity"].get(p.get("temperament_curiosity", ""), "—")
        return f"{ext} / {opt} / {cur}"

    cards = []
    for p in students:
        gender = p.get("gender", "")
        cls = "male" if gender == "male" else ("female" if gender == "female" else "")
        bg = (p.get("background", "") or "").strip()
        catch = p.get("catchphrase", "") or ""
        cards.append(
            f'<div class="persona-card {cls}">'
            f'<div class="pname">{html.escape(p.get("name","?"))}（{p.get("age","?")}歳・{gender}）</div>'
            f'<div class="pmeta">気質: {html.escape(_temp_str(p))}</div>'
            f'<div style="margin-top:0.3em;color:#444;font-size:0.92em">背景: {html.escape(bg[:220])}</div>'
            + (f'<div style="margin-top:0.2em;color:#555;font-size:0.92em">口癖: 「{html.escape(catch[:80])}」</div>' if catch else '')
            + f'</div>'
        )
    students_html = legend_html + '<div class="personas-grid">' + "".join(cards) + '</div>'

    sato_html = ""
    if sato:
        catch = sato.get("catchphrase", "") or ""
        sato_html = (
            '<h3 style="margin-top:1.5em">触媒（条件Bのみ）</h3>'
            '<div class="personas-grid">'
            f'<div class="persona-card sato">'
            f'<div class="pname">{html.escape(sato.get("name",""))}（{sato.get("age","?")}歳・{sato.get("gender","")}）</div>'
            f'<div class="pmeta">気質: {html.escape(_temp_str(sato))}</div>'
            f'<div style="margin-top:0.3em;color:#444;font-size:0.92em">背景: {html.escape((sato.get("background","") or "")[:380])}</div>'
            + (f'<div style="margin-top:0.2em;color:#555;font-size:0.92em">口癖: 「{html.escape(catch[:80])}」</div>' if catch else '')
            + '</div>'
            '</div>'
        )

    return students_html + sato_html


def stats_block_html(d: dict) -> str:
    msgs = d["msgs"]
    sender_count = collections.Counter(m["from_name"] for m in msgs)
    avg_len = sum(len(m.get("message", "")) for m in msgs) / max(1, len(msgs))
    top5 = "<br>".join(f"  {n}: {k}" for n, k in sender_count.most_common(5))
    return (
        f"メッセージ総数: <b>{len(msgs)}</b><br>"
        f"思考記録: <b>{len(d['mr'])}</b><br>"
        f"avg msg長: <b>{avg_len:.0f}</b>字<br>"
        f"<small>top 5 senders:<br>{top5}</small>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=str, help="condition A run dir")
    ap.add_argument("run_b", type=str, help="condition B run dir")
    ap.add_argument("--log-a", type=str, default=None, help="condition A sim log file (for token usage)")
    ap.add_argument("--log-b", type=str, default=None, help="condition B sim log file")
    ap.add_argument("--out", type=str, default=None, help="output HTML path (default: run_b/ab_comparison.html)")
    args = ap.parse_args()

    A = load_run(Path(args.run_a))
    B = load_run(Path(args.run_b))

    out_path = Path(args.out) if args.out else (Path(args.run_b) / "ab_comparison.html")

    # 分析側 (このスクリプト内) の token 消費を logging 経由で捕捉
    analysis_buf = io.StringIO()
    analysis_handler = logging.StreamHandler(analysis_buf)
    analysis_handler.setLevel(logging.INFO)
    analysis_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("llm_client_factory").addHandler(analysis_handler)
    logging.getLogger("llm_client_factory").setLevel(logging.INFO)
    analysis_t_start = time.time()

    # 触媒情報を取得 (Sato / UMA / 他)。B から取得 (A 条件には触媒なし)
    catalyst_b = get_catalyst_info(B)

    # Match participants: by axis_id (触媒 axis_id を除く)
    students_a = {p["axis_id"]: p for p in A["personas"] if is_participant(p, catalyst_b["axis_id"])}
    students_b = {p["axis_id"]: p for p in B["personas"] if is_participant(p, catalyst_b["axis_id"])}
    common_ids = sorted(set(students_a) & set(students_b))
    print(f"[info] common participants: {len(common_ids)}, catalyst: {catalyst_b['key']} ({catalyst_b['name']})")

    client = GeminiClient(
        model="gemini-3.1-flash-lite-preview", temperature=0.5, max_tokens=1500,
        enable_cache=False, enable_structured_output=False,
    )

    # Per participant: 2 calls (A, B)
    per_results: dict[str, dict] = {}

    def analyze(axis_id: str, cond: str):
        d = A if cond == "A" else B
        p = (students_a if cond == "A" else students_b)[axis_id]
        thoughts, sent, recv = per_agent_history(d, p["id"])
        prompt = build_analysis_prompt(p, thoughts, sent, recv, cond, catalyst_name=catalyst_b["name"])
        text = call_gemini(client, SYSTEM_ANALYSIS, prompt, max_tokens=1500)
        return axis_id, cond, text or "(LLM応答なし)"

    print("[info] analyzing each student in A and B...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = []
        for axis_id in common_ids:
            futures.append(ex.submit(analyze, axis_id, "A"))
            futures.append(ex.submit(analyze, axis_id, "B"))
        for fut in as_completed(futures):
            axis_id, cond, text = fut.result()
            per_results.setdefault(axis_id, {})[cond] = text
            print(f"  [ok] {axis_id} {cond} ({len(text)}chars)")

    # Synthesis
    print("[info] generating synthesis...")
    synth_blocks = []
    for axis_id in common_ids:
        p = students_a[axis_id]
        ext = p.get("temperament_extroversion", "?")
        opt = p.get("temperament_optimism", "?")
        cur = p.get("temperament_curiosity", "?")
        synth_blocks.append(
            f"--- {p.get('name')} ({axis_id}: 気質 ext={ext}/opt={opt}/cur={cur}) ---\n"
            f"[A] {per_results[axis_id].get('A','')}\n\n"
            f"[B] {per_results[axis_id].get('B','')}"
        )
    # 触媒の宛先集中度を計算してsynth promptに渡す
    cat_stats = catalyst_addressing_stats(B)
    quant_block_lines = ["## 定量データ (絶対に矛盾しないこと)"]
    quant_block_lines.append(f"- A条件 全メッセージ数: {len(A.get('msgs', []))}")
    quant_block_lines.append(f"- B条件 全メッセージ数: {len(B.get('msgs', []))}")
    cat_name = catalyst_b["name"]
    if cat_stats.get("present"):
        quant_block_lines.append(
            f"- {cat_name}の発話総数 (B): {cat_stats['total']} 回"
        )
        quant_block_lines.append(
            f"- 参加者が{cat_name}に向けて発話した回数: {cat_stats.get('addressed_to_catalyst', 0)} 回"
        )
        if cat_stats['total']:
            quant_block_lines.append(
                f"- {cat_name}の発話のうち broadcast 形式 (どの参加者にも宛先指定なし): {cat_stats['broadcast']} 回"
            )
            ranked = cat_stats.get("ranked") or []
            if ranked:
                rank_str = ", ".join([f"{nm} {c}回 ({c*100/cat_stats['total']:.0f}%)" for nm, c in ranked[:5]])
                quant_block_lines.append(f"- {cat_name}の発話の宛先別集計: {rank_str}")
                quant_block_lines.append(
                    f"- **{cat_name}の発話の最頻宛先: {cat_stats['top_name']} ({cat_stats['top_count']}回 / {cat_stats['top_share']*100:.0f}%)**"
                )
                quant_block_lines.append(
                    f"- {cat_name}が話しかけた相手のユニーク数: {cat_stats['unique_recipients']} 人 (10人中)"
                )
        else:
            # 発話しない触媒 (UMA等) — 別軸の指標を出す
            quant_block_lines.append(
                f"- ({cat_name}は発話しない触媒。代わりに参加者の言及量・トピック化度で判断する)"
            )

    # 触媒の波及度 (キーフレーズ伝染追跡 — 発話する触媒のみ)
    propagation = None
    if catalyst_b["speaks"]:
        print(f"[info] computing catalyst topic propagation ({cat_name})...")
        propagation = catalyst_topic_propagation(B, client=client)
        if propagation.get("phrases"):
            quant_block_lines.append("")
            quant_block_lines.append("## 触媒の中身がどれだけ集団に波及したか (key phrase tracking)")
            quant_block_lines.append(
                f"{cat_name}の発話に含まれた特徴フレーズが、{cat_name}の最初の使用step以降に他者の発話/思考に出現した回数。"
                "発話側=他者→他者の発話、思考側=他者のmemory/reasoning。"
                f"**人物名としての『{cat_name}』への言及ではなく、{cat_name}の発話の中身そのものが他者に取り込まれたかを測る指標**。"
            )
            for r in propagation["phrases"][:15]:
                spk = ", ".join([f"{nm}×{c}" for nm, c in r["msg_speakers"][:3]]) or "なし"
                thk = ", ".join([f"{nm}×{c}" for nm, c in r["mr_thinkers"][:3]]) or "なし"
                quant_block_lines.append(
                    f"- 「{r['phrase']}」(導入step{r['intro_step']}): 他者発話 {r['msg_count']}件 [{spk}] / 他者思考 {r['mr_count']}件 [{thk}]"
                )
            deep_pickup = sum(1 for r in propagation["phrases"] if r["msg_count"] >= 2)
            quant_block_lines.append(
                f"- **発話レベルで集団に2件以上拡散したフレーズ: {deep_pickup} / {len(propagation['phrases'])}個** "
                f"(発話レベル拡散率 {deep_pickup*100/max(1,len(propagation['phrases'])):.0f}%)"
            )
    else:
        # 発話しない触媒: 名前トピック化の度合いを集計
        print(f"[info] computing catalyst topic mention frequency ({cat_name}, non-speaking)...")
        b_msgs = B.get("msgs", [])
        b_mr = B.get("mr", [])
        # 触媒関連キーワード (固有名 + よくある呼称)
        kws = [cat_name, "謎の存在", "存在", "不思議", "観察", "動かない"]
        msg_mentions: dict[str, int] = collections.Counter()
        for m in b_msgs:
            text = m.get("message") or ""
            for k in kws:
                if k in text:
                    msg_mentions[k] += 1
        # 言及した参加者数
        speakers_who_mentioned = set()
        for m in b_msgs:
            text = m.get("message") or ""
            if any(k in text for k in [cat_name, "謎の存在", "存在", "不思議"]):
                speakers_who_mentioned.add(m.get("from"))
        # axis_id="UMA" は除外
        cat_id = catalyst_b["persona"]["id"]
        speakers_who_mentioned.discard(cat_id)
        quant_block_lines.append("")
        quant_block_lines.append("## 触媒のトピック化度 (名前/類語の言及量、発話しない触媒用)")
        for k, n in msg_mentions.most_common():
            quant_block_lines.append(f"- 「{k}」: messages.jsonl で {n}回")
        quant_block_lines.append(
            f"- {cat_name} 関連語に **言及した参加者の人数: {len(speakers_who_mentioned)} / {len(common_ids)}人**"
        )

    quant_block = "\n".join(quant_block_lines)

    system_synth = build_synthesis_prompt(catalyst_b)
    synth_user = quant_block + "\n\n" + "\n\n".join(synth_blocks)
    synth = call_gemini(client, system_synth, synth_user, max_tokens=2400) or "(synthesis失敗)"

    # HTML
    per_html_parts = []
    for axis_id in common_ids:
        p = students_a[axis_id]
        a_txt = per_results[axis_id].get("A", "")
        b_txt = per_results[axis_id].get("B", "")
        ext = TEMPERAMENT_LABELS["extroversion"].get(p.get("temperament_extroversion",""), "—")
        opt = TEMPERAMENT_LABELS["optimism"].get(p.get("temperament_optimism",""), "—")
        cur = TEMPERAMENT_LABELS["curiosity"].get(p.get("temperament_curiosity",""), "—")
        per_html_parts.append(
            f'<div class="agent-block">\n'
            f'  <div class="agent-name">{html.escape(p.get("name",""))}</div>\n'
            f'  <div class="agent-axis">{html.escape(axis_id)} · '
            f'気質: {html.escape(ext)} / {html.escape(opt)} / {html.escape(cur)}</div>\n'
            f'  <div class="cond-pair">\n'
            f'    <div class="cond-col A"><div class="cond-label A">条件A: 参加者のみ</div>{html.escape(a_txt)}</div>\n'
            f'    <div class="cond-col B"><div class="cond-label B">条件B: +{html.escape(catalyst_b["name"])}</div>{html.escape(b_txt)}</div>\n'
            f'  </div>\n</div>'
        )

    cfg_a = A["cfg"].get("metadata", {})
    cfg_b = B["cfg"].get("metadata", {})
    duration_steps = int(A["cfg"].get("simulation", {}).get("duration", 60))
    step_minutes = int(A["cfg"].get("simulation", {}).get("time_scale", {}).get("step_duration_minutes", 3))
    exp_meta = (
        f"参加者10人 × {duration_steps}ステップ ({step_minutes}分刻み, 約{duration_steps*step_minutes//60}時間相当)。"
        f"同一ペルソナ・同一seed で A/B を独立 run。"
        f"A: {cfg_a.get('description','')}。 B: {cfg_b.get('description','')}。"
    )

    # 触媒の発言と影響 (B のみ)。発話しない触媒は render_sato_block 内で別表示。
    sato_impacts = extract_sato_impact(B)
    sato_block_html = render_sato_block(sato_impacts, B)

    # 特徴的な出来事タイムライン (vertical alternating cards)
    events_a = extract_highlighted_events(A, "A")
    events_b = extract_highlighted_events(B, "B")
    timeline_a_html = render_highlight_timeline(events_a, "A", catalyst_b["name"])
    timeline_b_html = render_highlight_timeline(events_b, "B", catalyst_b["name"])
    # 全体活動タイムライン (interactive: slider + bubbles)
    interactive_a_html = build_interactive_timeline(A, "A")
    interactive_b_html = build_interactive_timeline(B, "B")

    # API 消費・所要時間
    sim_a_stats = parse_sim_log(Path(args.log_a)) if args.log_a else None
    sim_b_stats = parse_sim_log(Path(args.log_b)) if args.log_b else None
    analysis_dur = int(time.time() - analysis_t_start)
    analysis_stats = parse_logbuf(analysis_buf.getvalue())
    api_section_html = render_api_section(sim_a_stats, sim_b_stats, analysis_stats, analysis_dur)

    intro_section_html = render_intro_section(A, B)
    personas_section_html = render_personas_section(A, B)

    variant_key = detect_variant(A, B)
    vlabels = VARIANT_LABELS.get(variant_key, VARIANT_LABELS["high"])
    variant_title = vlabels["title"]
    variant_icon = vlabels["icon"]
    role = vlabels["role"]
    n_total = len([p for p in A["personas"] if is_participant(p, catalyst_b["axis_id"])])
    # Scene suffix: 旧 default の場所 (high/elem→教室, adult→アジト) のときは表示しない、
    # それ以外は「(公園版)」のように variant_title の右に付ける
    scene_label = get_scene_label(A, variant_key)
    default_label_for_variant = VARIANT_LABELS.get(variant_key, {}).get("scene", "")
    scene_suffix = f"（{scene_label}版）" if scene_label and scene_label != default_label_for_variant else ""
    duration_steps = int(A["cfg"].get("simulation", {}).get("duration", 60))
    step_minutes = int(A["cfg"].get("simulation", {}).get("time_scale", {}).get("step_duration_minutes", 3))
    variant_subtitle = f"{vlabels['role_full']} {n_total}人 × {duration_steps}step ({duration_steps*step_minutes}分相当) × A/B"

    out_html = HTML_TPL.format(
        variant_key=variant_key,
        variant_title=variant_title,
        variant_icon=variant_icon,
        variant_subtitle=variant_subtitle,
        scene_suffix=scene_suffix,
        catalyst_name=catalyst_b["name"],
        role=role,
        intro_section_html=intro_section_html,
        personas_section_html=personas_section_html,
        synth_html=html.escape(synth).replace("\n", "<br>"),
        stats_a_html=stats_block_html(A),
        stats_b_html=stats_block_html(B),
        sato_block_html=sato_block_html,
        interactive_a_html=interactive_a_html,
        interactive_b_html=interactive_b_html,
        timeline_a_html=timeline_a_html,
        timeline_b_html=timeline_b_html,
        per_student_html="\n".join(per_html_parts),
        api_section_html=api_section_html,
    )
    out_path.write_text(out_html, encoding="utf-8")
    print(f"[ok] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
