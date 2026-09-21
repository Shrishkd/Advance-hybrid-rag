"""Prompt loading — templates as versioned config, not string literals.

WHY THE PROMPT LIVES IN A FILE
------------------------------
Nothing in this project is pre-decided: every architectural choice is earned.
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


def load_template(name: str = "answer_v0",
                  required: tuple[str, ...] = REQUIRED) -> str:
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

    missing = [p for p in required if p not in body]
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


# Phrases a model uses when narrating its own reasoning rather than answering.
# Observed verbatim from qwen3:4b, which reasons regardless of think=False or
# /no_think and returns the trace UNTAGGED inside the answer, e.g.
#   "We are given a single source [S1]... We are to cite [S1]..."
#   "Hmm, the user is asking about..."
_REASONING_OPENERS = (
    # From the diagnostic prompt (2026-09-21):
    "we are given", "we are asked", "we need to", "we are to",
    "hmm", "okay, let", "okay, so", "alright, let",
    "the user is asking", "the user wants", "<think>",
    # From qwen3:4b under the REAL answer_v0 prompt - the first version of this
    # list missed all 14 of these, because it was built from the diagnostic
    # prompt's phrasing. First person, not first person plural:
    #   "I need to answer the question about..."
    #   "Let me carefully review the sources..."
    #   "Let me analyze the question and the sources..."
    "i need to", "i must", "i should", "i'll ", "i will ",
    "let me", "let's ", "first, i", "first, let",
)


def leaked_reasoning(answer: str) -> bool:
    """Does the answer open by narrating its own reasoning?

    WHY THIS CHECK EXISTS
    ---------------------
    The citation verifier asks "does every [S<n>] resolve?". A leaked
    reasoning trace passes that test: "We are to cite [S1]" cites S1. So a
    model that dumps its monologue into the answer scores as GROUNDED while
    producing something no user should see. Citation integrity and format
    integrity are different properties and need different checks.

    This is a HEURISTIC on the opening words, not a classifier. It will miss a
    trace that opens unusually and could flag a legitimate answer that happens
    to start "We need to..." - so it is reported as its own column, never
    folded silently into `grounded`.

    >>> leaked_reasoning("We are given a single source [S1]. The question asks...")
    True
    >>> leaked_reasoning("Hmm, the user is asking about regularization.")
    True
    >>> leaked_reasoning("I need to answer the question about Levenshtein.")
    True
    >>> leaked_reasoning("Let me carefully review the sources to find it.")
    True
    >>> leaked_reasoning("Regularization penalises large weights [S1].")
    False
    """
    head = answer.lstrip().lower()[:60]
    return any(head.startswith(o) for o in _REASONING_OPENERS)
