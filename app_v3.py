from flask import Flask, render_template, request, jsonify, session
from uuid import uuid4
from datetime import datetime
from bm25_version3 import (
    BM25DocumentRetriever,
    generate_prompt,
    call_qwen,
    llm_rerank_relevant_passages
)

app = Flask(__name__)
app.secret_key = 'ntu-secret-key-123'

# 初始化 BM25
retriever = BM25DocumentRetriever(pdf_folder="./ntu_rules_pdfs", corpus_path="bm25_docs_big.json")
retriever.build_or_load_corpus()
retriever.build_index()

# ---------- helpers ----------
def _now_str():
    return datetime.now().strftime("%Y/%m/%d %H:%M")

def _ensure_store():
    """
    session['store'] 結構：
    {
      'active_cid': 'xxxx',
      'conversations': {
         '<cid>': {'title': '新對話', 'created_at': '...', 'history': [ {'role':'user','content':'...'}, ... ] }
      }
    }
    """
    if "store" not in session or not isinstance(session["store"], dict):
        session["store"] = {"active_cid": None, "conversations": {}}
        _create_conversation(set_active=True)

def _create_conversation(set_active=True):
    cid = uuid4().hex[:8]
    conv = {"title": "新對話", "created_at": _now_str(), "history": []}
    session["store"]["conversations"][cid] = conv
    if set_active:
        session["store"]["active_cid"] = cid
    session.modified = True
    return cid

def _pair_history(history, max_pairs=10):
    pairs, cur_user = [], None
    for msg in history:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "user":
            cur_user = content
        elif role == "assistant" and cur_user is not None:
            pairs.append({"user": cur_user, "assistant": content})
            cur_user = None
    return pairs[-max_pairs:] if max_pairs and len(pairs) > max_pairs else pairs

# ---------- pages ----------
@app.route("/")
def index():
    _ensure_store()
    return render_template("index.html")

# ---------- APIs: conversations ----------
@app.route("/api/conversations", methods=["GET"])
def list_conversations():
    _ensure_store()
    convs = []
    for cid, c in session["store"]["conversations"].items():
        last = c["history"][-1]["content"] if c["history"] else ""
        convs.append({
            "id": cid,
            "title": c["title"],
            "created_at": c["created_at"],
            "last": last
        })
    return jsonify({
        "active": session["store"]["active_cid"],
        "items": convs
    })

@app.route("/api/conversations", methods=["POST"])
def new_conversation():
    _ensure_store()
    cid = _create_conversation(set_active=True)
    return jsonify({"id": cid})

@app.route("/api/conversations/<cid>", methods=["GET"])
def get_conversation(cid):
    _ensure_store()
    conv = session["store"]["conversations"].get(cid)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    session["store"]["active_cid"] = cid
    return jsonify({
        "id": cid,
        "title": conv["title"],
        "history": conv["history"]
    })

@app.route("/api/conversations/<cid>/title", methods=["POST"])
def rename_conversation(cid):
    _ensure_store()
    conv = session["store"]["conversations"].get(cid)
    if not conv:
        return jsonify({"error": "Conversation not found"}), 404
    title = (request.json or {}).get("title", "").strip()
    if title:
        conv["title"] = title
        session.modified = True
    return jsonify({"ok": True, "title": conv["title"]})

# ---------- chat ----------
@app.route("/ask", methods=["POST"])
def ask():
    _ensure_store()
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    cid = data.get("cid") or session["store"]["active_cid"]

    if not user_input:
        return jsonify({"response": "⚠️ 空訊息"}), 200
    if cid not in session["store"]["conversations"]:
        return jsonify({"error": "Conversation not found"}), 404

    conv = session["store"]["conversations"][cid]
    history = conv["history"]

    # 第一句自動當標題（15字內）
    if len(history) == 0:
        conv["title"] = user_input[:15]

    try:
        bm25_context = retriever.build_context(user_input, k=5, max_chars_per_doc=2000)
    except Exception:
        bm25_context = ""

    # ===== 新增：LLM 條文選段階段 =====
    refined_context = llm_rerank_relevant_passages(user_input, bm25_context)

    conversation_history = _pair_history(history, max_pairs=10)
    prompt = generate_prompt(user_input, refined_context, conversation_history)
    reply = call_qwen(prompt)

    # 更新對話歷史
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    session.modified = True

    return jsonify({
                "response": reply,
                "cid": cid,
                "bm25_titles": [r["doc_id"] for r in retriever.search(user_input, k=5)],
                "refined_context": refined_context
            })

@app.route("/api/conversations/<cid>", methods=["DELETE"])
def delete_conversation(cid):
    _ensure_store()
    convs = session["store"]["conversations"]
    if cid not in convs:
        return jsonify({"error": "Conversation not found"}), 404

    del convs[cid]
    if session["store"]["active_cid"] == cid:
        new_active = next(iter(convs), None)
        if new_active is None:
            new_active = _create_conversation(set_active=True)
        else:
            session["store"]["active_cid"] = new_active

    session.modified = True
    return jsonify({"ok": True, "active": session["store"]["active_cid"]})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=7777)