"""Citation faithfulness — the check the free verifier cannot do.

TWO QUESTIONS, TWO TOOLS
------------------------
    src/generate/citations.py   does [S3] RESOLVE to a supplied source?  free
    this file                   does [S3] SUPPORT what the answer says?  judge

The first is arithmetic. The second needs a model to read the source and the
claim side by side, which is why it lives behind a cloud judge and runs only
where it earns its cost.

The D9 result that forced this: llama3.2:3b scored 50/50 on every judge-free
check. A perfect score on checks that cannot see support is not evidence of
support. A terse model citing [S1] after a sentence S1 does not contain passes
everything in citations.py.

JUDGE != GENERATOR
------------------
The judge should not be the generator's own model (self-preference bias). The judge
here is gpt-oss:120b-cloud. That is clean for llama and phi4, and SAME FAMILY
for gpt-oss:20b-cloud - so that row carries a caveat, and a same-family win
must not be read as independent evidence.

SCORING
-------
Per answer, the judge returns a verdict per distinct citation. We report:

    supported_rate    fraction of citations judged "supported"
    any_unsupported   did the answer contain at least one bad citation?

`partial` counts as half. The per-answer flag matters more than the rate for a
user: one invented attribution in an otherwise good answer is still an answer
you cannot trust.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.generate.citations import extract_citations

VERDICT_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


@dataclass
class FaithReport:
    verdicts: dict[str, str] = field(default_factory=dict)   # sid -> verdict
    parse_failed: bool = False

    @property
    def n(self) -> int:
        return len(self.verdicts)

    @property
    def supported_rate(self) -> float | None:
        if not self.verdicts:
            return None
        return sum(VERDICT_SCORE.get(v, 0.0) for v in self.verdicts.values()) / self.n

    @property
    def any_unsupported(self) -> bool:
        return any(v == "unsupported" for v in self.verdicts.values())


def parse_verdicts(raw: str, expected: list[str]) -> FaithReport:
    """Pull verdicts out of the judge's reply, tolerating stray prose.

    Citations the judge SKIPPED are recorded as "unsupported", not dropped. A
    judge that quietly omits the hard case would otherwise make the rate look
    better the less it audited.

    >>> r = parse_verdicts('{"verdicts":[{"sid":"S1","verdict":"supported"}]}',
    ...                    ["S1", "S2"])
    >>> r.verdicts
    {'S1': 'supported', 'S2': 'unsupported'}
    >>> r.supported_rate, r.any_unsupported
    (0.5, True)
    >>> parse_verdicts("no json here", ["S1"]).parse_failed
    True
    """
    m = re.search(r"\{.*\}", raw, re.S)
    got: dict[str, str] = {}
    if m:
        try:
            for v in json.loads(m.group(0)).get("verdicts", []):
                sid = str(v.get("sid", "")).upper().strip("[]")
                verdict = str(v.get("verdict", "")).lower().strip()
                if sid and verdict in VERDICT_SCORE:
                    got[sid] = verdict
        except (json.JSONDecodeError, AttributeError):
            return FaithReport(parse_failed=True)
    else:
        return FaithReport(parse_failed=True)
    return FaithReport({sid: got.get(sid, "unsupported") for sid in expected})


def judge_answer(client, judge_model: str, template: str, question: str,
                 answer: str, sources: list[dict]) -> FaithReport | None:
    """Judge one answer. Returns None when there is nothing to judge.

    Only the sources the answer actually CITED are shown to the judge. Showing
    all six would invite it to find support in an uncited chunk and mark the
    citation "supported" for the wrong reason.
    """
    cited, _ = extract_citations(answer)
    by_sid = {s["sid"].upper(): s for s in sources}
    cited = [c for c in cited if c in by_sid]    # fabricated ids: citations.py's job
    if not cited:
        return None
    src_block = "\n\n".join(f"[{c}] {by_sid[c]['text']}" for c in cited)
    prompt = (template.replace("{question}", question)
                      .replace("{answer}", answer)
                      .replace("{sources}", src_block))
    resp = client.chat(judge_model, prompt, max_tokens=1500, temperature=0.0)
    return parse_verdicts(resp.text or "", cited)
