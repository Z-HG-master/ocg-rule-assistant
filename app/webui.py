# -*- coding: utf-8 -*-
"""检索测试台：① 纯检索召回测试 ② AI 测试（检索注入后调用对话 API）。"""
import json
import time

from flask import Flask, Response, jsonify, request

from . import chat_client, config
from .retrieve import build_context, retrieve

app = Flask(__name__)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>OCG 规则助手 · 检索测试台</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f6f7f9;color:#222}
header{background:#fff;border-bottom:1px solid #e3e6ea;padding:16px 24px}
h1{font-size:18px;margin:0 0 2px}p.sub{color:#666;margin:0;font-size:13px}
.tabs{display:flex;gap:8px;margin:14px 0 0}
.tab{padding:8px 16px;border-radius:6px;background:#eef1f5;cursor:pointer;font-size:14px}
.tab.on{background:#2f6fed;color:#fff}
main{padding:20px 24px;max-width:1100px}
.row{display:flex;gap:8px}input{flex:1;padding:10px 12px;font-size:15px;border:1px solid #ccd;border-radius:6px}
button{padding:10px 18px;font-size:15px;border:0;border-radius:6px;background:#2f6fed;color:#fff;cursor:pointer}
button:disabled{background:#9ab}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:14px 16px;margin-top:14px}
.card h2{font-size:15px;margin:0 0 10px}
.tag{font-size:12px;padding:2px 8px;border-radius:99px;background:#f0f1f4;color:#888;margin-right:6px}
.tag.on{background:#e6f7ec;color:#1c7a45}
.muted{color:#777;font-size:13px}.lane{font-size:12px;color:#3550b8;background:#eef2ff;border-radius:4px;padding:1px 6px;margin-right:6px}
.block{border-top:1px dashed #e3e6ea;padding:10px 0}.block:first-child{border-top:0;padding-top:0}
pre{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font-size:13px}
.hide{display:none}
</style></head><body>
<header><h1>OCG 规则助手 · 检索测试台 <span class="muted" style="font-size:12px">v1.1</span></h1>
<p class="sub">三个库：卡片库 · 规则库 · 裁定库；裁定召回为父块/卡片/向量三路加权 RRF。</p>
<div class="tabs"><div class="tab on" id="tabR">检索召回测试</div><div class="tab" id="tabA">AI 测试</div></div></header>
<main>
<div class="row"><input id="q" placeholder="例如：效果被无效的怪兽能否作为效果的对象？"><button id="go">开始</button></div>
<div id="out"></div>
</main>
<script>
var mode="retrieve";
var input=document.getElementById("q"), btn=document.getElementById("go"), out=document.getElementById("out");
var tabR=document.getElementById("tabR"), tabA=document.getElementById("tabA");
function setMode(m){mode=m;tabR.className="tab"+(m==="retrieve"?" on":"");tabA.className="tab"+(m==="ai"?" on":"");out.innerHTML="";}
tabR.addEventListener("click",function(){setMode("retrieve")});
tabA.addEventListener("click",function(){setMode("ai")});
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function tag(name,on){return '<span class="tag '+(on?"on":"")+'">'+name+(on?" ✓":" ·")+"</span>";}
btn.addEventListener("click",go);input.addEventListener("keydown",function(e){if(e.key==="Enter")go();});
function go(){
 var q=input.value.trim(); if(!q)return;
 btn.disabled=true;btn.textContent=mode==="ai"?"生成中…":"检索中…";out.innerHTML="";
 var url=mode==="ai"?"/api/ask":"/api/retrieve";
 fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:q})})
  .then(function(r){return r.json();})
  .then(function(d){ mode==="ai"?renderAsk(d):renderRetrieve(d); })
  .catch(function(e){ out.innerHTML='<div class="card">出错：'+esc(e.message)+"</div>"; })
  .then(function(){ btn.disabled=false;btn.textContent="开始"; });
}
function renderRetrieve(d){
 var h=[];var r=d.result;
 h.push('<div class="card"><h2>路由概览 <span class="muted">'+d.ms+' ms</span></h2>');
 h.push("<div>"+tag("卡片库",r.cards.length>0)+tag("规则库",r.rules.blocks.length>0)+tag("裁定库",!!r.rulings.picked)+"</div>");
 h.push("<div style='margin-top:10px'>命中卡片："+r.cards.length+" 张</div>");
 h.push("<div>规则父块："+(r.rules.parents.length?esc(r.rules.parents.join("、")):"（无）")+"</div>");
 h.push("<div>裁定父块："+(r.rulings.parents.length?esc(r.rulings.parents.join("、")):"（无）")+"</div>");
 h.push("<div>三路候选："+r.rulings.lanes.map(function(x){return '<span class="lane">'+x[0]+" "+x[1]+"</span>";}).join("")+"</div></div>");
 h.push('<div class="card"><h2>卡片库</h2>');
 if(r.cards.length){r.cards.forEach(function(c){
   h.push('<div class="block"><b>'+esc(c.name)+'</b> <span class="muted">cid '+c.cid+"</span><pre>"+esc(c.text)+"</pre>");
   if(c.supplement){h.push('<div class="muted" style="margin-top:6px">补充说明（官方补充情报）</div><pre>'+esc(c.supplement)+"</pre>");}
   h.push("</div>");});}
 else h.push('<div class="muted">未召回卡片</div>');
 h.push("</div>");
 h.push('<div class="card"><h2>规则库</h2>');
 if(r.rules.blocks.length){r.rules.blocks.forEach(function(b){h.push('<div class="block"><div class="muted">'+esc(b.bid)+"</div><pre>"+esc(b.display)+"</pre></div>");});}
 else h.push('<div class="muted">未召回规则</div>');
 h.push("</div>");
 h.push('<div class="card"><h2>裁定库</h2>');
 if(r.rulings.picked){var p=r.rulings.picked;
  h.push('<div class="block"><div class="muted">'+esc(p.id)+" · RRF "+p.score+"</div>");
  h.push("<div>"+p.lanes.map(function(x){return '<span class="lane">'+x[0]+" 第 "+x[1]+"</span>";}).join("")+"</div>");
  h.push("<pre>"+esc(p.question)+"\n\n"+esc(p.conclusion)+"</pre></div>");}
 else h.push('<div class="muted">未召回裁定</div>');
 h.push("</div>");
 out.innerHTML=h.join("");
}
function renderAsk(d){
 var h=[];
 h.push('<div class="card"><h2>回答 <span class="muted">'+d.ms+' ms · '+esc(d.model)+"</span></h2><pre>"+esc(d.answer)+"</pre></div>");
 h.push('<div class="card"><h2>注入的资料</h2><pre>'+esc(d.context)+"</pre></div>");
 out.innerHTML=h.join("");
}
</script></body></html>"""


@app.route("/")
def index():
    resp = Response(PAGE, mimetype="text/html; charset=utf-8")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    t0 = time.time()
    q = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    return jsonify({"ms": int((time.time() - t0) * 1000), "result": retrieve(q)})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    t0 = time.time()
    q = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not q:
        return jsonify({"error": "empty query"}), 400
    r = retrieve(q)
    ctx = build_context(r)
    messages = [
        {"role": "system", "content": "你是游戏王 OCG 规则裁判助手，回答要给出明确结论和简短理由。"},
        {"role": "user", "content": ctx + "\n\n用户问题：" + q},
    ]
    answer = chat_client.chat(messages)
    return jsonify({"ms": int((time.time() - t0) * 1000), "model": config.get("chat.model"),
                    "answer": answer, "context": ctx})


if __name__ == "__main__":
    app.run(host=config.get("server.host", "127.0.0.1"),
            port=int(config.get("server.port", 7871)), threaded=True)
