import json, re
from typing import Literal, List, Dict

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# ─────────── 0.  MODEL HANDLES ──────────────────────────────
def make_llm(temp: float) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="",
        model="gpt-4o",
        api_key="aa-",
        temperature=temp,
    )

LLM_PRECISE = make_llm(0.0)   # Phase A
LLM_DEEP    = make_llm(0.7)   # Phase B

MAX_TOKENS = 512

# ─────────── 1-A.  PRECISION PROMPT  (Phase A) ─────────────
PRECISION_TMPL = """
You are the Privacy Gate.

Rewrite the text so that its vagueness matches ε = {epsilon}.

Guidelines
• ε = 1.0  → keep almost all specifics; redact only obvious secrets
• ε = 0.7  → generalise a little
• ε = 0.5  → remove/blur most specifics
• ε = 0.3  → keep only broad actions
• ε = 0.1  → bare outline

Do NOT remove the core meaning.

Label: {label}
Original text:
\"\"\"{chunk}\"\"\"

Return JSON only:
{{"rewritten": "<sanitised>"}}"""

# ─────────── 1-B.  DEEP-OBFUSCATE PROMPT (Phase B) ─────────
DEEP_TMPL = """
You are the Privacy Gate – deep-obfuscation mode.

Take the input text and rewrite it so it is **even vaguer** than before:
• shorten sentences,
• replace any remaining specifics with generic terms,
• rephrase with **different wording** than the previous version,
• keep overall intent.

Return JSON only:
{{"rewritten": "<even more vague text>"}}"""

# ─────────── 2.  LOW-LEVEL CALL HELPERS ─────────────────────
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _call(llm: ChatOpenAI, prompt: str) -> str:
    resp = llm(
        [SystemMessage(content="You are a privacy-focused assistant."),
         HumanMessage(content=prompt)],
        max_tokens=MAX_TOKENS,
    ).content.strip()
    m = _JSON_RE.search(resp)
    if not m:
        raise ValueError(f"Missing JSON:\n{resp}")
    obj = json.loads(m.group(0))
    if "rewritten" not in obj:
        raise ValueError(f"No 'rewritten' key:\n{obj}")
    return obj["rewritten"]

# ─────────── 3-A.  single-pass sanitiser (Phase A) ─────────
def sanitise_once(chunk: str, label: str, ε: float) -> str:
    prompt = PRECISION_TMPL.format(
        label=label, epsilon=ε, chunk=chunk.replace('"', '\\"')
    )
    return _call(LLM_PRECISE, prompt)

# ─────────── 3-B.  deep-obfuscate step (Phase B) ───────────
def deep_obfuscate(prev: str) -> str:
    prompt = DEEP_TMPL.format(chunk=prev.replace('"', '\\"'))
    return _call(LLM_DEEP, prompt)

# ─────────── 4.  public pipeline ───────────────────────────
def privacy_gate_pipeline(
    text: str,
    label: Literal["PUBLIC", "SENSITIVE", "CONFIDENTIAL"],
    eps_schedule: List[float] = (1.0, 0.7, 0.5, 0.3, 0.1),
    deep_rounds: int = 4,
) -> Dict[float, List[str]]:
    """
    Phase A: run once for every ε in eps_schedule
    Phase B: take ε=min(eps_schedule) output, run 'deep_rounds' extra passes
    Returns {ε: [v1, v2, …]}  (ε>min → one element; ε_min → deep_rounds+1 elems)
    """
    results = {}
    current = text
    for ε in eps_schedule:
        current = sanitise_once(current, label, ε)
        results[ε] = [current]

    ε_min = min(eps_schedule)
    for _ in range(deep_rounds):
        current = deep_obfuscate(current)
        results[ε_min].append(current)

    return results
