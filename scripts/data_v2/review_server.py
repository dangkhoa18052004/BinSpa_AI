"""Owner review page on localhost (selfie_v1). Non-expert questions only: one image per item, three buttons.

A task is a folder reports/selfie_v1/review/<task>/ with items.json:
  {"title": ..., "question": ..., "help": ..., "items": [{"id": ..., "image": "<file in the folder>", "note": ...}]}
Answers are appended to datasets/selfie_v1/reviews/<task>.jsonl as {"id", "answer": yes|no|unsure, "utc"};
the latest answer per item counts. Nothing leaves the machine; binds to 127.0.0.1.
Usage: venv/Scripts/python.exe scripts/data_v2/review_server.py --task kbm0q_mapping [--port 8010]
"""
import argparse
import json
import time
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / 'reports/selfie_v1/review'
ANSWERS = ROOT / 'datasets/selfie_v1/reviews'

PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duyệt ảnh</title><style>
:root{--bg:#fafafa;--fg:#1d1d1f;--muted:#666;--card:#fff;--line:#ddd;--yes:#1e7d32;--no:#b3261e;--unsure:#8a6d00}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#aaa;--card:#1f1f1f;--line:#333}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}.muted{color:var(--muted);font-size:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px}
img{max-width:100%;height:auto;display:block;margin:8px auto;border-radius:4px}
.btns{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:8px}
button{font:inherit;padding:10px 22px;border-radius:6px;border:0;color:#fff;cursor:pointer}
.yes{background:var(--yes)}.no{background:var(--no)}.unsure{background:var(--unsure)}
.nav{background:transparent;color:var(--fg);border:1px solid var(--line)}
.done{font-weight:600}</style></head><body><main>
<h1 id="title"></h1><div class="muted" id="help"></div>
<div class="card"><div id="pos" class="muted"></div><div id="q" class="done"></div><img id="img" alt="ảnh cần duyệt">
<div id="note" class="muted"></div>
<div class="btns"><button class="yes" onclick="send('yes')">Đúng (1)</button><button class="no" onclick="send('no')">Sai (2)</button>
<button class="unsure" onclick="send('unsure')">Không chắc (3)</button></div>
<div class="btns"><button class="nav" onclick="go(-1)">← Trước</button><button class="nav" onclick="go(1)">Sau →</button></div></div>
<p class="muted" id="status"></p></main><script>
let T,i=0,ans={};
async function load(){T=await (await fetch('items')).json();ans=await (await fetch('answers')).json();
document.getElementById('title').textContent=T.title;document.getElementById('help').textContent=T.help||'';
document.getElementById('q').textContent=T.question;i=T.items.findIndex(x=>!(x.id in ans));if(i<0)i=0;show()}
function show(){const it=T.items[i];document.getElementById('img').src='img/'+encodeURIComponent(it.image);
document.getElementById('note').textContent=it.note||'';const n=Object.keys(ans).length;
document.getElementById('pos').textContent=`Ảnh ${i+1}/${T.items.length} · đã duyệt ${n}`+(ans[it.id]?` · đã chọn: ${{yes:'Đúng',no:'Sai',unsure:'Không chắc'}[ans[it.id]]}`:'');
document.getElementById('status').textContent=n>=T.items.length?'Đã duyệt xong tất cả. Có thể đóng trang.':''}
async function send(a){const it=T.items[i];await fetch('answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:it.id,answer:a})});
ans[it.id]=a;if(i<T.items.length-1)i++;show()}
function go(d){i=Math.min(Math.max(i+d,0),T.items.length-1);show()}
document.addEventListener('keydown',e=>{if(e.key==='1')send('yes');if(e.key==='2')send('no');if(e.key==='3')send('unsure');
if(e.key==='ArrowLeft')go(-1);if(e.key==='ArrowRight')go(1)});load()</script></body></html>"""


def create_app(task):
    folder = TASKS / task
    items = json.loads((folder / 'items.json').read_text(encoding='utf-8'))
    ids = {it['id'] for it in items['items']}
    store = ANSWERS / f'{task}.jsonl'
    app = Flask(__name__)
    app.json.ensure_ascii = False

    @app.get('/')
    def page():
        return PAGE

    @app.get('/items')
    def get_items():
        return jsonify(items)

    @app.get('/img/<path:name>')
    def img(name):
        return send_from_directory(folder, name)

    @app.get('/answers')
    def answers():
        out = {}
        if store.exists():
            for line in store.read_text(encoding='utf-8').splitlines():
                r = json.loads(line)
                out[r['id']] = r['answer']
        return jsonify(out)

    @app.post('/answer')
    def answer():
        r = request.get_json(force=True)
        if r.get('id') not in ids or r.get('answer') not in ('yes', 'no', 'unsure'):
            abort(400)
        store.parent.mkdir(parents=True, exist_ok=True)
        with store.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(id=r['id'], answer=r['answer'], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')
        return jsonify(ok=True)

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', required=True)
    p.add_argument('--port', type=int, default=8010)
    a = p.parse_args()
    print(f'Mở trình duyệt: http://127.0.0.1:{a.port}  (Ctrl+C để dừng)', flush=True)
    create_app(a.task).run(host='127.0.0.1', port=a.port, use_reloader=False)


if __name__ == '__main__':
    main()
