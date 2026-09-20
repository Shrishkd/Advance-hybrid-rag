"""Prompt loading — templates as versioned config, not string literals.

WHY THE PROMPT LIVES IN A FILE
------------------------------
`CLAUDE.md`: "Nothing is pre-decided... every architectural choice is earned."
A prompt is an architectural choice. Buried in a Python string it becomes
invisible, unversioned and untestable; as `configs/prompts/answer_v<n>.txt` it
is a config value that can be swapped, diffed and benchmarked exactly like
`chunk_size` or `retrieval.mode`.

That also makes prompt ITERATION an experiment rather than a habit. Copy
answer_v0.txt to answer_v1.txt, change one instruction, run the lab, compare.
One variable at a time applies to prompts too.

COMMENTS ARE STRIPPED
---------------------
Template files carry a '#' comment header explaining which parts are
load-bearing. Those lines are removed before the prompt reaches a model:
instructions meant for the human editing the file would otherwise become
instructions to the generator.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path("configs/prompts")

REQUIRED = ("{context}", "{question}")

# The exact string the template tells the model to emit when the sources do
# not answer the question. Kept here, next to the loader, because the
# evaluation layer must look for the SAME token the prompt asks for - if these
# two ever drift, refusals silently stop being detected.
REFUSAL_TOKEN = "NOT_IN_SOURCES"


def load_template(name: str = "answer_v0") -> str:
    """Read a prompt template, stripping its comment header.

    Raises if a required placeholder is missing. That check is worth having:
    a renamed placeholder produces a KeyError deep inside a generation run,
    after models have loaded and time has been spent.
    """
    path = PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        have = sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))
        raise FileNotFoundError(f"no prompt {name!r} in {PROMPT_DIR}; have {have}")

    body = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ).strip()

    missing = [p for p in REQUIRED if p not in body]
    if missing:
        raise ValueError(f"{path} is missing placeholder(s): {missing}")
    return body


def build_prompt(question: str, context: str, template: str) -> str:
    """Fill a template.

    `str.replace` rather than `str.format`: chunk text is full of braces -
    LaTeX, code, set notation like {0,1} - and `format` would raise or
    misinterpret them. The corpus is mathematics textbooks; this is not a
    hypothetical.

    >>> t = "Q: {question}\\nC: {context}"
    >>> build_prompt("what is x?", "p(x) = {0,1}", t)
    'Q: what is x?\\nC: p(x) = {0,1}'
    """
    return template.replace("{question}", question).replace("{context}", context)


def is_refusal(answer: str) -> bool:
    """Did the model decline, as the template instructs?

    Checked as a prefix-ish containment rather than equality: models reliably
    emit the token and then explain, which the template explicitly asks for.

    >>> is_refusal("NOT_IN_SOURCES The passage does not give a BLEU score.")
    True
    >>> is_refusal("  not_in_sources - nothing here about that.")
    True
    >>> is_refusal("The answer is 0.42 [S1].")
    False
    """
    return REFUSAL_TOKEN.lower() in answer.lower()
