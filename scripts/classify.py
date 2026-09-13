import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT_PATH = Path(__file__).parent.parent
scope, prompt_set, model = sys.argv[1:4]
base_url = sys.argv[4] if len(sys.argv) > 4 else None

PROMPTS = {
    "bare": '''Is the following text a folktale?

Respond with only this JSON, no other text:
{
  "reasoning": "<one line>",
  "is_folktale": true/false
}

Text:
"""
%s
"""''',
    "simple": '''Is the following text a folktale - a traditional story of the kind passed down
orally, with archetypal characters, formulaic language, and a simple moral or
magical plot?

Respond with only this JSON, no other text:
{
  "reasoning": "<one line>",
  "is_folktale": true/false
}

Text:
"""
%s
"""''',
    "detailed": '''You are given an article from "Kmetijske in rokodelske novice", a 19th-century
Slovenian agricultural newspaper. Decide whether the article is a folktale in
its own right.

Count as a folktale only a NARRATED traditional story - the article itself
tells the story, with agents, sequenced events and a resolution, and the plot
comes from tradition rather than the author's invention. This includes wonder
tales (pravljica), local and etiological legends (povedka), tales of
mythological beings such as vile or povodni mož (bajka), myths (mit), saints'
legends with miracle narration (legenda), animal fables with a moral (basen),
and parables retold from tradition (prilika) - whether transcribed from a
teller, retold by a named author, or translated from another tradition.

Do NOT count:
- proverbs, sayings, riddles, or descriptions of superstitions - no narrative
- folk songs and lyric verse; verse counts only if it narrates a traditional
  story, and an attributed poet's own composition never counts
- jokes and short jocular anecdotes
- an article that merely QUOTES, EMBEDS, or DISCUSSES folk material inside a
  longer report, essay, or scholarly piece - the article itself is then not a
  folktale
- news, history, biography, practical farming advice, sermons, and fiction
  invented by its author, however story-like

Most articles in this newspaper are farming advice, news and correspondence;
genuine folktales are rare. Answer true only on positive evidence of a narrated
traditional story, not because the text is old, rural, or moralising.

Respond with only this JSON, no other text:
{
  "reasoning": "<one line>",
  "is_folktale": true/false
}

Text:
"""
%s
"""''',
}

HEAD, TAIL = 70_000, 30_000
FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
VERDICT_RE = re.compile(r'"is_folktale"\s*:\s*(true|false)', re.IGNORECASE)
REASON_RE = re.compile(r'"reasoning"\s*:\s*"(.*?)"\s*[,}\n]', re.DOTALL)

def build_prompt(text):
    if len(text) > HEAD + TAIL:
        text = text[:HEAD] + "\n\n[... elided ...]\n\n" + text[-TAIL:]

    return PROMPTS[prompt_set] % text

def parse_verdict(raw):
    body = FENCE_RE.sub("", raw.strip()).strip()
    start, end = body.find("{"), body.rfind("}")

    if start != -1 and end > start:
        try:
            payload = json.loads(body[start:end + 1])
            if isinstance(payload, dict) and "is_folktale" in payload:
                return {"is_folktale": bool(payload["is_folktale"]),
                        "reasoning": str(payload.get("reasoning") or ""),
                        "malformed_json": False}
        except json.JSONDecodeError:
            pass

    match = VERDICT_RE.search(body)

    if not match:
        raise ValueError("no is_folktale verdict in reply")
    
    reason = REASON_RE.search(body)

    return {"is_folktale": match.group(1).lower() == "true",
            "reasoning": (reason.group(1) if reason else "").strip(),
            "malformed_json": True}


def call(prompt, budget):
    if base_url:
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=budget, temperature=0)
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("incomplete response (length)")
        return response.choices[0].message.content or ""
    
    response = client.responses.create(model=model, input=prompt, max_output_tokens=budget, reasoning={"effort": "none"}, temperature=0)
    if response.status == "incomplete":
        raise RuntimeError("incomplete response")
    
    return response.output_text or ""


def classify(urn):
    row = {"article_urn": urn, "model": model, "prompt": prompt_set}
    budget = 1500

    for attempt in range(4):
        try:
            return {**row, "status": "ok", **parse_verdict(call(build_prompt(texts[urn]), budget))}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            budget = min(budget * 2, 8000)
            time.sleep(2 ** attempt)

    return {**row, "status": "error", "error": error}


texts = {a["urn"]: a["text"] for a in json.load(open(ROOT_PATH / "articles.json", encoding="utf-8"))}

if scope == "gold":
    urns = [g["urn"] for g in json.load(open(ROOT_PATH / "gold" / "gold.json", encoding="utf-8"))]
else:
    urns = sorted(texts)

client = OpenAI(api_key="EMPTY", base_url=base_url) if base_url else OpenAI(api_key=os.environ["OPENAI_API_KEY"])
slug = re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-")
out = ROOT_PATH / "eval" / f"{scope}_{prompt_set}_{slug}.jsonl"
out.parent.mkdir(exist_ok=True)
done = {json.loads(line)["article_urn"] for line in open(out, encoding="utf-8")} if out.exists() else set()
todo = [u for u in urns if u not in done]

print(f"{out.name}: {len(done)} done, {len(todo)} to classify")

with ThreadPoolExecutor(8) as pool, open(out, "a", encoding="utf-8") as fh:
    for row in tqdm(pool.map(classify, todo), total=len(todo)):
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
