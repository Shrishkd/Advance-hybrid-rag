"""Generate a LOCAL html review page for curating golden-set candidates.

    python -m evaluation.make_review_page
    # then open data/golden/review.html in a browser

WHY LOCAL, NOT A HOSTED PAGE
----------------------------
The page embeds excerpts of copyrighted textbook text, and the corpus
never leaves this machine, so this writes a plain file opened with
file:// - nothing is uploaded, and it works offline.

WORKFLOW
--------
    review in browser -> Export -> save over data/golden/candidates.jsonl
    -> python -m evaluation.curate --promote

Decisions autosave to localStorage as you go, so closing the tab does not lose
work. Export writes the full JSONL back with `keep` set and any edits applied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CANDIDATES = Path("data/golden/candidates.jsonl")
OUT = Path("data/golden/review.html")

CSS = """
:root{--bg:#fbfbfd;--fg:#1c1c1e;--mut:#6b6b70;--line:#e3e3e8;--card:#fff;
--keep:#1a7f37;--rej:#c4314b;--accent:#4a3aff;--warn:#9a6700}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#141416;--fg:#ececf1;--mut:#9a9aa2;--line:#2c2c31;--card:#1c1c20;
--keep:#3fb950;--rej:#ff6b81;--accent:#8b7cff;--warn:#d29922}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:12px 16px;z-index:9}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:var(--accent);transition:width .2s}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.pill{padding:2px 9px;border-radius:99px;border:1px solid var(--line);font-size:12px}
.k{color:var(--keep)}.r{color:var(--rej)}.u{color:var(--mut)}
main{max-width:900px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:18px;margin-bottom:14px}
label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.05em;
color:var(--mut);margin:14px 0 5px}
textarea,input,select{width:100%;background:transparent;color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:9px;font:inherit}
textarea{resize:vertical}
.src{background:var(--bg);border:1px solid var(--line);border-radius:8px;
padding:10px;margin-top:8px;font-size:13px}
.src b{color:var(--accent);font-weight:600}
.ex{color:var(--mut);font-size:12.5px;white-space:pre-wrap;max-height:150px;
overflow:auto;margin-top:6px}
button{font:inherit;padding:9px 16px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}
button.keep{border-color:var(--keep);color:var(--keep)}
button.rej{border-color:var(--rej);color:var(--rej)}
.hint{color:var(--mut);font-size:12.5px}
kbd{border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px;
background:var(--bg);font-family:ui-monospace,monospace}
.warn{color:var(--warn);font-size:12.5px;margin-top:6px}
"""

# Kept as a plain concatenated string (no f-strings, no .format) because the JS
# is full of ${...} template literals and {} braces that would otherwise need
# escaping.
JS = r"""
const RAW = JSON.parse(document.getElementById('data').textContent);
const KEY = 'golden-review-v1';
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { saved = {}; }
const D = RAW.map(r => Object.assign({}, r, saved[r.qid] || {}));
let i = 0, filter = '';

const types = [...new Set(D.map(d => d.question_type))].sort();
const fs = document.getElementById('filt');
types.forEach(t => fs.insertAdjacentHTML('beforeend', '<option value="' + t + '">' + t + '</option>'));
fs.onchange = e => { filter = e.target.value; i = 0; render(); };

const view = () => D.map((d, idx) => ({ d, idx })).filter(x => !filter || x.d.question_type === filter);

function persist() {
  const out = {};
  D.forEach(d => {
    out[d.qid] = { keep: d.keep, question: d.question,
      ground_truth_answer: d.ground_truth_answer, question_type: d.question_type,
      difficulty: d.difficulty, notes: d.notes };
  });
  try { localStorage.setItem(KEY, JSON.stringify(out)); } catch (e) {}
}
function set(idx, field, val) { D[idx][field] = val; persist(); stats(); }
function decide(v) {
  const V = view(); if (!V.length) return;
  D[V[i].idx].keep = v; persist();
  if (i < V.length - 1) i++;
  render();
}
function stats() {
  const k = D.filter(d => d.keep === true).length, r = D.filter(d => d.keep === false).length;
  document.getElementById('nk').textContent = k;
  document.getElementById('nr').textContent = r;
  document.getElementById('nu').textContent = D.length - k - r;
  document.getElementById('prog').style.width = (100 * (k + r) / D.length) + '%';
}
const esc = s => String(s == null ? '' : s).replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function render() {
  const V = view();
  if (!V.length) { document.getElementById('app').innerHTML = '<div class="card">no items</div>'; return; }
  if (i >= V.length) i = V.length - 1;
  const d = V[i].d, idx = V[i].idx;
  document.getElementById('pos').textContent = (i + 1) + ' / ' + V.length;
  const mark = d.keep === true ? '<span class="k">KEEP</span>'
             : d.keep === false ? '<span class="r">REJECT</span>'
             : '<span class="u">undecided</span>';
  const ctxs = d.ground_truth_contexts || [];
  const ctx = ctxs.map(c => c.book + ' p' + c.page_start + '-' + c.page_end).join('  ');
  const paired = ctxs.length > 1;
  const secs = d._source_sections || [], exs = d._source_excerpt || [];
  let src = '';
  for (let n = 0; n < secs.length; n++) {
    src += '<div class="src"><b>' + esc(ctxs[n] ? ctxs[n].book : '') + '</b> ' +
           esc(secs[n]) + '<div class="ex">' + esc(exs[n] || '') + '</div></div>';
  }
  let typeOpts = '';
  types.forEach(t => { typeOpts += '<option ' + (t === d.question_type ? 'selected' : '') + '>' + t + '</option>'; });
  let diffOpts = '';
  ['easy', 'medium', 'hard', 'unanswerable'].forEach(t => {
    diffOpts += '<option ' + (t === d.difficulty ? 'selected' : '') + '>' + t + '</option>';
  });

  document.getElementById('app').innerHTML =
  '<div class="card">' +
    '<div class="row"><b>' + esc(d.qid) + '</b> ' + mark +
      '<span class="pill">' + esc(d.question_type) + '</span>' +
      '<span class="pill">' + esc(d.difficulty) + '</span>' +
      '<span class="pill">' + esc(ctx) + '</span></div>' +
    (paired ? '<div class="warn">PAIRED &mdash; check: could passage 1 alone answer this? If yes, it is not multi-hop.</div>' : '') +
    '<label>question</label>' +
    '<textarea rows="2" oninput="set(' + idx + ',\'question\',this.value)">' + esc(d.question) + '</textarea>' +
    '<label>reference answer</label>' +
    '<textarea rows="4" oninput="set(' + idx + ',\'ground_truth_answer\',this.value)">' + esc(d.ground_truth_answer) + '</textarea>' +
    '<div class="row">' +
      '<div style="flex:1"><label>type</label><select onchange="set(' + idx + ',\'question_type\',this.value)">' + typeOpts + '</select></div>' +
      '<div style="flex:1"><label>difficulty</label><select onchange="set(' + idx + ',\'difficulty\',this.value)">' + diffOpts + '</select></div>' +
    '</div>' +
    '<label>notes</label>' +
    '<input value="' + esc(d.notes) + '" oninput="set(' + idx + ',\'notes\',this.value)">' +
    '<label>source passages (read-only)</label>' + src +
    '<div class="row" style="margin-top:16px">' +
      '<button class="keep" onclick="decide(true)">Keep (K)</button>' +
      '<button class="rej" onclick="decide(false)">Reject (R)</button>' +
      '<button onclick="i=Math.max(0,i-1);render()">&larr;</button>' +
      '<button onclick="i=Math.min(view().length-1,i+1);render()">&rarr;</button>' +
    '</div>' +
  '</div>';
  stats();
}
document.addEventListener('keydown', e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === 'k' || e.key === 'K') decide(true);
  else if (e.key === 'r' || e.key === 'R') decide(false);
  else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); render(); }
  else if (e.key === 'ArrowRight') { i = Math.min(view().length - 1, i + 1); render(); }
});
function exportJsonl() {
  const lines = D.map(d => JSON.stringify(d));
  const blob = new Blob([lines.join('\n') + '\n'], { type: 'application/jsonl' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'candidates.jsonl';
  a.click();
}
render();
"""

PAGE = (
    '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "<title>Golden Set Review</title>\n<style>" + CSS + "</style></head><body>\n"
    "<header>\n <div class=\"row\">\n"
    "  <b>Golden Set Review</b>\n"
    '  <span id="pos" class="pill"></span>\n'
    '  <span class="pill k">keep <b id="nk">0</b></span>\n'
    '  <span class="pill r">reject <b id="nr">0</b></span>\n'
    '  <span class="pill u">left <b id="nu">0</b></span>\n'
    '  <select id="filt" style="width:auto"><option value="">all types</option></select>\n'
    '  <button onclick="exportJsonl()">Export JSONL</button>\n'
    " </div>\n"
    ' <div class="bar"><div id="prog" style="width:0"></div></div>\n'
    ' <div class="hint" style="margin-top:7px">'
    "<kbd>K</kbd> keep &nbsp; <kbd>R</kbd> reject &nbsp; "
    "<kbd>&larr;</kbd><kbd>&rarr;</kbd> navigate &nbsp; edits save automatically</div>\n"
    "</header>\n"
    '<main><div id="app"></div></main>\n'
    '<script id="data" type="application/json">__DATA__</script>\n'
    "<script>" + JS + "</script></body></html>\n"
)


def main() -> int:
    if not CANDIDATES.exists():
        print(f"missing {CANDIDATES} - run evaluation.generate_candidates first")
        return 1
    recs = [
        json.loads(l)
        for l in CANDIDATES.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    # "</" inside the embedded JSON would close the <script> tag early.
    blob = json.dumps(recs, ensure_ascii=False).replace("</", "<\\/")
    OUT.write_text(PAGE.replace("__DATA__", blob), encoding="utf-8")
    print(f"wrote {OUT}  ({len(recs)} candidates)")
    print(f"open: {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
