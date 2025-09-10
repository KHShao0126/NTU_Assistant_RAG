let activeCid = null;
let typingMsg = null;

// 🔒 併發控制 / 防亂序
let inflight = false;
let lastReqId = 0;

// ---------- UI helpers ----------
function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}
function showTyping() {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = el("div", "chat-message assistant typing");
  const bubble = el("div", "bubble");
  const spinner = el("div", "spinner");
  const tip = el("span", "typing-text");
  tip.textContent = "思考中…";
  bubble.appendChild(spinner);
  bubble.appendChild(tip);
  msgDiv.appendChild(bubble);
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  typingMsg = msgDiv;
}
function removeTyping() {
  if (typingMsg && typingMsg.parentNode) typingMsg.parentNode.removeChild(typingMsg);
  typingMsg = null;
}
function setSendingState(isSending) {
  const btn = document.getElementById("send-btn");
  const input = document.getElementById("user-input");
  if (btn) {
    btn.disabled = isSending;
    btn.classList.toggle("is-sending", isSending);
  }
  if (input) input.disabled = isSending; // 避免輸入期間又送出
}
function appendMessage(role, content) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = el("div", "chat-message " + role);
  const bubble = el("div", "bubble");
  bubble.textContent = content;
  msgDiv.appendChild(bubble);
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}
function clearChat() {
  document.getElementById("chat-box").innerHTML = "";
}

// ---------- conversations ----------
async function refreshConvList() {
  const res = await fetch("/api/conversations");
  const data = await res.json();
  const list = document.getElementById("conv-list");
  list.innerHTML = "";

  data.items.forEach(item => {
    const isActive = (item.id === activeCid);
    const row = el("div", "conv-item" + (isActive ? " active" : ""));
    const title = el("div", "conv-title");  title.textContent = item.title || "新對話";
    const preview = el("div", "conv-preview"); preview.textContent = item.last || "";

    // 👉 垃圾桶
    const actions = el("div", "conv-actions");
    const delBtn = el("button", "icon-btn danger");
    delBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM7 9h2v9H7V9z"></path>
      </svg>`;
    delBtn.title = "刪除對話";
    delBtn.onclick = (e) => { e.stopPropagation(); deleteConversation(item.id); };
    actions.appendChild(delBtn);

    row.appendChild(title);
    row.appendChild(preview);
    row.appendChild(actions);

    row.onclick = () => openConversation(item.id);
    list.appendChild(row);
  });
}

async function deleteConversation(cid) {
  const ok = confirm("確定要刪除此對話嗎？此動作無法復原。");
  if (!ok) return;

  const res = await fetch(`/api/conversations/${cid}`, { method: "DELETE" });
  const data = await res.json();

  // 重新整理列表並切到後端回傳的 active（或自動新建的那個）
  await refreshConvList();
  if (data.active) {
    activeCid = data.active;
    await openConversation(activeCid);
  } else {
    await newConversation();
  }
}

async function openConversation(cid) {
  if (inflight) return; // 避免正在送出時切換
  const res = await fetch(`/api/conversations/${cid}`);
  if (!res.ok) return;
  const data = await res.json();

  // ✅ 先設定 activeCid
  activeCid = data.id;

  // render history
  clearChat();
  (data.history || []).forEach(msg => appendMessage(msg.role, msg.content));

  // ✅ 再更新列表 → 正確加上 active 樣式
  await refreshConvList();
}

async function newConversation() {
  if (inflight) return;
  const res = await fetch("/api/conversations", { method: "POST" });
  const data = await res.json();
  await openConversation(data.id);
}

// ---------- chat ----------
async function sendMessage() {
  const inputBox = document.getElementById("user-input");
  const message = (inputBox.value || "").trim();

  if (!message || !activeCid || inflight) return;

  inflight = true;                 // 🔒 上鎖
  const myReq = ++lastReqId;       // ⏱️ 這次請求的編號

  appendMessage("user", message);
  inputBox.value = "";
  setSendingState(true);
  showTyping();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, cid: activeCid, reqId: myReq })
    });
    const data = await res.json();

    // ❗如果不是最新請求、或回的不是當前對話，直接丟棄避免汙染
    if (myReq !== lastReqId || !data || (data.cid && data.cid !== activeCid)) {
      removeTyping();
      return;
    }

    removeTyping();
    appendMessage("assistant", data.response || "⚠️ 回答失敗");
    refreshConvList(); // 更新側邊預覽 & active 樣式
  } catch (e) {
    removeTyping();
    appendMessage("assistant", "⚠️ 連線失敗");
  } finally {
    inflight = false;              // 🔓 解鎖
    setSendingState(false);
    inputBox.focus();
  }
}

// ---------- boot ----------
window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("send-btn").addEventListener("click", sendMessage);
  document.getElementById("user-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
    if (inflight) e.stopPropagation(); // 輸入上鎖時避免再送
  });
  document.getElementById("new-chat").addEventListener("click", newConversation);

  await refreshConvList();
  if (!activeCid) await newConversation();
  else await openConversation(activeCid);
});