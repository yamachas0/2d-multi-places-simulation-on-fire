"""Add 3-dim temperament to existing classroom persona YAMLs via Gemini.

For each persona, infer:
  - extroversion: 内向的 / mid / 社交的
  - optimism:     心配性 / mid / 楽天的
  - curiosity:    慎重 / mid / 好奇心旺盛

Reads existing fields (name, occupation, family_struct, recent_concern,
catchphrase, etc.) and asks Gemini to assign one of {high, mid, low} per
dimension based on the persona's apparent character.

Usage:
  python tools/add_temperament.py --variant adult
  python tools/add_temperament.py --variant high
  python tools/add_temperament.py --variant elementary
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from llm_client_factory import GeminiClient

VARIANT_PERSONA_PATH = {
    "high":       ROOT / "classroom_personas.yaml",
    "adult":      ROOT / "classroom_personas_adult.yaml",
    "elementary": ROOT / "classroom_personas_elementary.yaml",
}

SYSTEM = (
    "あなたは人物の気質を3つの軸でラベル付けする評価者です。\n"
    "各軸は high / mid / low で答えます (3軸とも独立)。\n"
    "出力は JSON のみ。各人物につき "
    "{\"axis_id\": \"<axis_id>\", \"extroversion\": \"high|mid|low\", "
    "\"optimism\": \"high|mid|low\", \"curiosity\": \"high|mid|low\"} 形式の配列。\n"
    "軸の意味:\n"
    "- extroversion: high=社交的(他者と関わるのが好き) / low=内向的(一人を好む) / mid=どちらでもない\n"
    "- optimism:     high=楽天的(物事を前向きに見る) / low=心配性(慎重・不安が先に立つ) / mid=どちらでもない\n"
    "- curiosity:    high=好奇心旺盛(新しいものに飛びつく) / low=慎重(知らないことには距離を取る) / mid=どちらでもない\n"
    "ペルソナの記述から見える性格をそのまま読んで判断する。原則として 3軸が全部同じ値にならないよう、人物ごとに違いが出るように。"
)


def build_user_prompt(personas: list[dict]) -> str:
    lines = ["以下の人物それぞれについて、3軸を判定してJSON配列で返す。"]
    for p in personas:
        keys = ["axis_id", "name", "age", "occupation", "family_struct",
                "join_reason", "recent_concern", "catchphrase",
                "club", "subjects", "future_interest", "phone_habit",
                "grade", "hobby_like", "hobby_dislike"]
        block = [f"---", f"axis_id: {p.get('axis_id', '?')}", f"name: {p.get('name','')}"]
        for k in keys:
            v = p.get(k)
            if v and k not in ("axis_id", "name"):
                block.append(f"{k}: {v}")
        lines.append("\n".join(block))
    lines.append("---")
    lines.append("JSON配列のみで返答してください。前置き・コードブロック禁止。")
    return "\n".join(lines)


def call_gemini(personas: list[dict]) -> list[dict]:
    client = GeminiClient(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.3,
        max_tokens=2000,
        enable_cache=False,
        enable_structured_output=False,
    )
    user_prompt = build_user_prompt(personas)
    response = client.generate(SYSTEM, user_prompt)
    # Extract JSON array
    m = re.search(r"\[\s*\{.*?\}\s*\]", response, re.S)
    if not m:
        raise RuntimeError(f"Could not find JSON array in response: {response[:300]}")
    return json.loads(m.group(0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_PERSONA_PATH))
    args = ap.parse_args()

    path = VARIANT_PERSONA_PATH[args.variant]
    if not path.exists():
        print(f"[err] missing {path}", file=sys.stderr)
        return 1
    personas = yaml.safe_load(path.read_text(encoding="utf-8"))
    print(f"[info] loaded {len(personas)} personas from {path.name}")

    print("[info] calling Gemini for temperament inference...")
    results = call_gemini(personas)
    print(f"[info] received {len(results)} entries")

    by_axis = {r["axis_id"]: r for r in results}
    for p in personas:
        r = by_axis.get(p.get("axis_id"))
        if not r:
            print(f"[warn] no result for {p.get('axis_id')}")
            continue
        p["temperament_extroversion"] = r.get("extroversion", "mid")
        p["temperament_optimism"]     = r.get("optimism",     "mid")
        p["temperament_curiosity"]    = r.get("curiosity",    "mid")

    path.write_text(
        yaml.safe_dump(personas, allow_unicode=True, sort_keys=False, default_flow_style=False, width=4096),
        encoding="utf-8",
    )
    print(f"[ok] wrote {path}")
    print()
    print("=== assigned temperament ===")
    for p in personas:
        e = p.get("temperament_extroversion", "?")
        o = p.get("temperament_optimism", "?")
        c = p.get("temperament_curiosity", "?")
        print(f"  {p.get('axis_id'):4} {p.get('name'):14}  ext={e:4} opt={o:4} cur={c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
