"""Build the blind labelling page for Phase 8 judge validation.

    python -m evaluation.make_judge_labels      # writes data/golden/judge_labels.html
    # open it, label every citation, click "Export", save as
    # data/golden/judge_labels.jsonl, then:
    python -m evaluation.judge_agreement

WHY THIS IS REQUIRED, NOT OPTIONAL
----------------------------------
D9 chose gpt-oss:20b-cloud as the generator. The only free judge is gpt-oss:120b -
the SAME family - and CLAUDE.md forbids trusting a judge on its own family's output
without checking (self-preference bias). Shrish accepted the same-family judge on one
condition: validate it against human labels first.

THE DESIGN THAT MAKES THE CHECK MEAN SOMETHING
----------------------------------------------
- BLIND: the judge's verdicts are not on the page, so they cannot anchor the labels.
- A CONTROL GROUP: 8 answers from gpt-oss (same family as the judge - the case under
  test) and 7 from llama3.2:3b (different family). Self-preference shows up as the
  judge agreeing with the human LESS on gpt-oss answers, specifically by calling them
  "supported" when the human did not.
- SAME TASK AS THE JUDGE: per citation, does THIS source support the claim attached
  to it? Not "is the answer good".
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path("data/golden/judge_labels.html")
PICK = {"gpt-oss:20b-cloud": 8, "llama3.2:3b": 7}


def load(model: str) -> list[dict]:
    p = Path(f"reports/perq/generation_golden_{model.replace(':', '_')}.jsonl")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("faith_verdicts") and r.get("answer")]


def main() -> int:
    rng = random.Random(21)
    items = []
    for model, n in PICK.items():
        rows = load(model)
        rng.shuffle(rows)
        for r in rows[:n]:
            by_sid = {s["sid"].upper(): s for s in r["sources_json"]}
            items.append({
                "id": f"{model}|{r['qid']}",
                "question": r["question"],
                "answer": r["answer"],
                "citations": [
                    {"sid": sid, "cite": by_sid[sid]["cite"], "text": by_sid[sid]["text"]}
                    for sid in r["faith_verdicts"] if sid in by_sid
                ],
            })
    rng.shuffle(items)   # interleave models so the labeller cannot tell them apart

    data = json.dumps(items, ensure_ascii=False)
    html = PAGE.replace("__DATA__", data.replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    n_cites = sum(len(i["citations"]) for i in items)
    print(f"{len(items)} answers, {n_cites} citations to label -> {OUT}")
    return 0


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Judge validation — blind labels</title>
<style>
 body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:0 auto;padding:16px;
      background:#fafafa;color:#1a1a1a}
 .card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:14px 0}
 .q{font-weight:600;margin-bottom:6px}.a{background:#f3f6fb;padding:10px;border-radius:6px;
      white-space:pre-wrap}
 .cite{border-top:1px dashed #ccc;margin-top:12px;padding-top:10px}
 .src{font-size:13px;color:#444;background:#fffbea;padding:8px;border-radius:6px;
      max-height:220px;overflow:auto;white-space:pre-wrap}
 button{margin:6px 6px 0 0;padding:6px 12px;border:1px solid #999;border-radius:6px;
      background:#fff;cursor:pointer}
 button.on{background:#1a1a1a;color:#fff}
 #bar{position:sticky;top:0;background:#fafafa;padding:8px 0;border-bottom:1px solid #ddd}
 mark{background:#dbeafe}
</style></head><body>
<div id="bar"><b>Judge validation</b> — for each citation: does THIS source support the
claim the answer attaches to <mark>[S#]</mark>? Judge only from the source text.
<span id="prog"></span> <button onclick="exp()">Export JSONL</button></div>
<div id="list"></div>
<script>
const ITEMS = __DATA__;
const KEY = "judge_labels_v1";
let L = {}; try { L = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) {}
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function hi(t){ return esc(t).replace(/\\[(S\\d+)\\]/gi, '<mark>[$1]</mark>'); }
function set(k, v){ L[k] = v; try{localStorage.setItem(KEY, JSON.stringify(L));}catch(e){} draw(); }
function draw(){
  let total = 0, done = 0, h = "";
  ITEMS.forEach((it, i) => {
    h += `<div class="card"><div class="q">${i+1}. ${esc(it.question)}</div>`;
    h += `<div class="a">${hi(it.answer)}</div>`;
    it.citations.forEach(c => {
      const k = it.id + "|" + c.sid; total++; if (L[k]) done++;
      h += `<div class="cite"><b>[${c.sid}]</b> ${esc(c.cite)}<div class="src">${esc(c.text)}</div>`;
      ["supported","partial","unsupported"].forEach(v =>
        h += `<button class="${L[k]===v?'on':''}" onclick="set('${k.replace(/'/g,"\\\\'")}','${v}')">${v}</button>`);
      h += `</div>`;
    });
    h += `</div>`;
  });
  document.getElementById("list").innerHTML = h;
  document.getElementById("prog").textContent = ` · ${done}/${total} labelled`;
}
function exp(){
  const lines = Object.entries(L).map(([k, v]) => {
    const [model, qid, sid] = k.split("|");
    return JSON.stringify({model, qid, sid, label: v});
  });
  const b = new Blob([lines.join("\\n") + "\\n"], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(b);
  a.download = "judge_labels.jsonl"; a.click();
}
draw();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
