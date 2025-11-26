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
app.secret_key = 'ntu-secret-key-456'

# 初始化 BM25
retriever = BM25DocumentRetriever(pdf_folder="./ntu_rules_pdfs", corpus_path="bm25_docs_big.json")
retriever.build_or_load_corpus()
retriever.build_index()

from flask import send_from_directory

@app.route("/pdf/<path:filename>")
def serve_pdf(filename):
    return send_from_directory("ntu_rules_pdfs", filename)

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

def strip_document_header_and_link(text: str):
    """
    從 reranker 的 refined_context 中：
    1. 找出 [Document X] 後面的 PDF 檔名
    2. 去掉第一段 header，只回傳純法規原文
    3. 回傳 pdf_link (可用於後端 /pdf/<filename>)
    """

    import re

    if not text:
        return "", None

    text = text.strip()

    # 尋找形如：[Document 2] 資訊工程學系轉系相關規定.pdf
    m = re.search(r"\[Document\s+\d+\]\s*([^\n]+?\.pdf)", text)

    pdf_filename = None
    if m:
        pdf_filename = m.group(1).strip()

    # 去掉 header（[Document ...]那一行）
    m_header = re.search(r"\[Document\s+\d+\][^\n]*", text)
    if m_header:
        cleaned = text[m_header.end():].lstrip()
    else:
        cleaned = text

    return cleaned, pdf_filename

@app.route("/api/profile", methods=["GET"])
def get_profile():
    return jsonify(session.get("profile", {}))

@app.route("/api/profile", methods=["POST"])
def save_profile():
    data = request.get_json()
    session["profile"] = data
    session.modified = True
    return jsonify({"ok": True})

MAX_HISTORY_MSGS = 20          # 最多保留 20 則訊息
MAX_CONTENT_CHARS = 1500       # 每則訊息最多 1500 個字元

def _shrink_history(history):
    # 1. 只保留最後 N 則
    if len(history) > MAX_HISTORY_MSGS:
        del history[:-MAX_HISTORY_MSGS]

    # 2. 每則訊息太長就只留尾巴
    for msg in history:
        c = msg.get("content", "")
        if isinstance(c, str) and len(c) > MAX_CONTENT_CHARS:
            msg["content"] = c[-MAX_CONTENT_CHARS:]

# ---------- chat ----------
@app.route("/ask", methods=["POST"])
def ask():
    _ensure_store()
    data = request.get_json(silent=True) or {}
    user_input = (data.get("message") or "").strip()
    cid = data.get("cid") or session["store"]["active_cid"]

    # 讀取 k（容錯 + 夾在 [1, 20] 之間，自己可調上限）
    try:
        k = int(data.get("k", 5))
    except (TypeError, ValueError):
        k = 5
    k = max(1, min(20, k))

    if not user_input:
        return jsonify({"response": "⚠️ 空訊息"}), 200
    if cid not in session["store"]["conversations"]:
        return jsonify({"error": "Conversation not found"}), 404

    conv = session["store"]["conversations"][cid]
    history = conv["history"]

    if len(history) == 0:
        conv["title"] = user_input[:15]

    # ===== 這一段是新的：組合「學生背景 + 問題」給 BM25 用 =====
    profile = session.get("profile", {})  # 你在 /api/profile 存的那個

    year    = profile.get("year")    # e.g. "112"
    degree  = profile.get("degree")  # e.g. "學士"
    college = profile.get("college") # e.g. "電機資訊學院"
    dept    = profile.get("dept")    # e.g. "資訊工程學系"

    meta_parts = []
    if year:
        meta_parts.append(f"入學年份{year}年")
    if degree:
        meta_parts.append(f"{degree}")
    if college:
        meta_parts.append(college)
    if dept:
        meta_parts.append(dept)

    conv_pairs = _pair_history(history, max_pairs=5)

    # 這個字串專門給 BM25 用，問題 + 學生背景
    bm25_query = user_input
    if meta_parts:
        bm25_query = user_input + "；學生背景：" + "、".join(meta_parts) + "、".join([f"{p['user']}：{p['assistant']}" for p in conv_pairs])

    print("【BM25 查詢字串】", bm25_query)
    

    try:
        # ⬇️ 這裡用動態 k（原本是固定 5）
        bm25_context = retriever.build_context(bm25_query, k=k, max_chars_per_doc=2000)
    except Exception:
        bm25_context = ""

    # ===== LLM 條文選段階段（保持不變）=====
    refined_context = llm_rerank_relevant_passages(user_input, bm25_context)
    

    conversation_history = _pair_history(history, max_pairs=10)
    prompt = generate_prompt(user_input, refined_context, conversation_history)
    reply = call_qwen(prompt)
    clean_context, pdf_filename = strip_document_header_and_link(refined_context)

    if pdf_filename:
        pdf_link = f"/pdf/{pdf_filename}"
    else:
        pdf_link = None

    reply = (
        reply
        + "\n\n"
        + "━━━━━━━━━━ 📎 相關法規原文 ━━━━━━━━━━\n"
        + clean_context
        + f"\n\n📄 參考文件連結：{pdf_link}" or ""
    )

    # 更新歷史
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    _shrink_history(history)
    session.modified = True

    # 回傳時也用 k，讓前端能顯示「Top-k」
    return jsonify({
        "response": reply,
        "cid": cid,
        "k": k,
        "bm25_titles": [r["doc_id"] for r in retriever.search(user_input, k=k)],
        "refined_context": refined_context,
        "pdf_link": pdf_link
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
    app.run(debug=False, host='0.0.0.0', port=7777)