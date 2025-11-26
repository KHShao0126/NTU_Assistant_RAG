let activeCid = null;
let typingMsg = null;

// 併發控制
let inflight = false;
let lastReqId = 0;

// ---------- DOM helpers ----------
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }

function showTyping() {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = el("div", "chat-message assistant typing");
  const bubble = el("div", "bubble");
  const spinner = el("div", "spinner");
  const tip = el("span", "typing-text");
  tip.textContent = "思考中…";
  bubble.appendChild(spinner); bubble.appendChild(tip);
  msgDiv.appendChild(bubble); chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  typingMsg = msgDiv;
}
function removeTyping() { if (typingMsg && typingMsg.parentNode) typingMsg.parentNode.removeChild(typingMsg); typingMsg = null; }

// 把純文字變成可換行、含超連結的 HTML
function renderMessageHTML(content) {
  if (!content) return "";

  // 先做 HTML escape
  let safe = content
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // 把網址變成 <a>，含 http/https 或 /pdf 開頭的路徑
  safe = safe.replace(
    /(https?:\/\/[^\s]+|\/pdf\/[^\s]+\.pdf)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  // 換行變成 <br>
  safe = safe.replace(/\n/g, "<br>");

  return safe;
}

const COLLEGES = {
  "文學院": ["戲劇學系","中國文學系","外國語文學系","日本語文學系","歷史學系","人類學系","哲學系","圖書資訊學系"],
  "社會科學院": ["經濟學系","政治學系","社會學系","社會工作學系"],
  "理學院": ["心理學系","地理環境資源學系","化學系","地質科學系","物理學系","大氣科學系","數學系"],
  "管理學院": ["會計學系","工商管理學系","國際企業學系","財務金融學系","資訊管理學系"],
  "法律學院": ["法律學系"],
  "生命科學院": ["生命科學系","生化科技學系"],
  "生物資源暨農學院": [
    "農藝學系","生物機電工程學系","生物環境系統工程學系","動物科學技術學系",
    "園藝暨景觀學系","植物病理與微生物學系","農業經濟學系","生物產業傳播暨發展學系",
    "農業化學系","森林環境暨資源學系"
  ],
  "醫學院": ["醫學系","物理治療學系","職能治療學系","護理學系","醫學檢驗暨生物技術學系"],
  "電機資訊學院": ["電機工程學系","資訊工程學系","資訊網路與多媒體研究所","生醫電子與資訊學研究所"],
  "工學院": ["機械工程學系","土木工程學系","化學工程學系","材料科學與工程學系","應用力學研究所"],
  "獸醫專業學院": ["獸醫學系"],
  "公共衛生學院": ["公共衛生學系"]
};

function populateCollegeAndDept(selectedCollege = "", selectedDept = "") {
  const colSel = document.getElementById("p_college");
  const deptSel = document.getElementById("p_dept");
  if (!colSel || !deptSel) return;

  // 第一次開啟時，填學院
  if (!colSel.options.length) {
    Object.keys(COLLEGES).forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      colSel.appendChild(opt);
    });

    // 當學院改變時，重填學系
    colSel.addEventListener("change", () => {
      populateDeptOnly(colSel.value);
    });
  }

  // 設定目前學院
  if (selectedCollege && COLLEGES[selectedCollege]) {
    colSel.value = selectedCollege;
  } else if (!colSel.value) {
    colSel.value = Object.keys(COLLEGES)[0];
  }

  // 依學院填學系
  populateDeptOnly(colSel.value, selectedDept);
}

function populateDeptOnly(college, selectedDept = "") {
  const deptSel = document.getElementById("p_dept");
  if (!deptSel) return;
  deptSel.innerHTML = "";

  const list = COLLEGES[college] || [];
  list.forEach(d => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    deptSel.appendChild(opt);
  });

  if (selectedDept && list.includes(selectedDept)) {
    deptSel.value = selectedDept;
  } else if (list.length) {
    deptSel.value = list[0];
  }
}


function closeProfile(){
  document.getElementById("profileModal").style.display = "none";
}

async function loadProfile(){
  const res = await fetch("/api/profile");
  const data = await res.json();

  document.getElementById("p_year").value    = data.year || "112";
  document.getElementById("p_degree").value  = data.degree || "學士";
  populateCollegeAndDept(data.college, data.dept);

  document.getElementById("p_sid").value    = data.sid     || "";
}



function setSendingState(isSending) {
  const btn = document.getElementById("send-btn");
  const input = document.getElementById("user-input");
  const kRange = document.getElementById("topkRange");
  const kNumber = document.getElementById("topkNumber");
  const kInput = document.getElementById("kInput");

  if (btn) { btn.disabled = isSending; btn.classList.toggle("is-sending", isSending); }
  if (input) input.disabled = isSending;
  if (kRange) kRange.disabled = isSending;
  if (kNumber) kNumber.disabled = isSending;
  if (kInput) kInput.disabled = isSending;
}

function appendMessage(role, content, extraInfo = null) {
  const chatBox = document.getElementById("chat-box");
  const msgDiv = document.createElement("div");
  msgDiv.classList.add("chat-message", role);

  // 泡泡
  const bubble = document.createElement("div");
  bubble.classList.add("bubble");

  // 使用 innerHTML + renderMessageHTML，保留換行 & 超連結
  bubble.innerHTML = renderMessageHTML(content);

  msgDiv.appendChild(bubble);

  // 只有 assistant 才有「三個點」按鈕
  if (role === "assistant" && extraInfo) {
    const menuBtn = document.createElement("button");
    menuBtn.className = "msg-menu-btn";
    menuBtn.type = "button";
    menuBtn.innerHTML = "⋯";
    menuBtn.title = "查看參考資料（Top-K、擷取條文等）";

    menuBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      showReferences(extraInfo);
    });

    msgDiv.appendChild(menuBtn);
  }

  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function clearChat(){ document.getElementById("chat-box").innerHTML=""; }

// ---------- Top-K 同步與讀取 ----------
function clampK(v){ v = parseInt(v,10); if(isNaN(v)) v = 5; return Math.max(1, Math.min(50, v)); }

function syncTopK(from){
  const kRange = document.getElementById("topkRange");
  const kNumber = document.getElementById("topkNumber");
  const kInput = document.getElementById("kInput");

  let val;
  if(from === "range") val = clampK(kRange.value);
  else if(from === "number") val = clampK(kNumber.value);
  else if(from === "input") val = clampK(kInput.value);
  else val = clampK( (kNumber && kNumber.value) || (kRange && kRange.value) || (kInput && kInput.value) || 5 );

  if(kRange)  kRange.value  = val;
  if(kNumber) kNumber.value = val;
  if(kInput)  kInput.value  = val;   // 舊程式會讀它
}

function readTopK(){
  const kNumber = document.getElementById("topkNumber");
  const kRange  = document.getElementById("topkRange");
  const kInput  = document.getElementById("kInput");
  // 以 number 為準 → range → 備用 kInput
  const val = (kNumber && kNumber.value) || (kRange && kRange.value) || (kInput && kInput.value) || 5;
  return clampK(val);
}

// ---------- 參考資料彈窗 ----------
function showReferences(info) {
  const modal = document.getElementById("refModal");
  const contentDiv = document.getElementById("modalContent");
  let html = "<h4> 設定</h4>";
  html += `<div>Top-K：<b>${info.k ?? readTopK()}</b></div><hr/>`;
  html += "<h4> BM25 擷取文件：</h4><ul>";
  (info.bm25_titles || []).forEach(t => { html += `<li>${t}</li>`; });
  html += `</ul><hr><h4> 條文選段：</h4><pre style="white-space:pre-wrap;">${info.refined_context || "(無資料)"}</pre>`;
  contentDiv.innerHTML = html;
  modal.style.display = "flex";
}
function closeModal(){ document.getElementById("refModal").style.display="none"; }

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

    const actions = el("div", "conv-actions");
    const delBtn = el("button", "icon-btn danger");
    delBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16"><path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v9h-2V9zm4 0h2v9h-2V9zM7 9h2v9H7V9z"></path></svg>`;
    delBtn.title = "刪除對話";
    delBtn.onclick = (e) => { e.stopPropagation(); deleteConversation(item.id); };
    actions.appendChild(delBtn);

    row.appendChild(title); row.appendChild(preview); row.appendChild(actions);
    row.onclick = () => openConversation(item.id);
    list.appendChild(row);
  });
}

async function deleteConversation(cid){
  const ok = confirm("確定要刪除此對話嗎？此動作無法復原。");
  if(!ok) return;
  const res = await fetch(`/api/conversations/${cid}`, { method:"DELETE" });
  const data = await res.json();
  await refreshConvList();
  if(data.active){ activeCid = data.active; await openConversation(activeCid); }
  else { await newConversation(); }
}

async function openConversation(cid){
  if(inflight) return;
  const res = await fetch(`/api/conversations/${cid}`);
  if(!res.ok) return;
  const data = await res.json();
  activeCid = data.id;
  clearChat();
  // 🔽 切換對話時，把下方 PDF 連結清空
  const pdfBar = document.getElementById("pdf-bar");
  if (pdfBar) {
    pdfBar.style.display = "none";
    pdfBar.innerHTML = "";
  }
  (data.history || []).forEach(msg => appendMessage(msg.role, msg.content));
  await refreshConvList();
}

async function newConversation(){
  if(inflight) return;
  const res = await fetch("/api/conversations", { method:"POST" });
  const data = await res.json();
  await openConversation(data.id);
}

// ---------- chat ----------
async function sendMessage(){
  const inputBox = document.getElementById("user-input");
  const message = (inputBox.value || "").trim();
  if(!message || !activeCid || inflight) return;

  inflight = true;
  const myReq = ++lastReqId;
  syncTopK();                 // 先確保三者同步
  const topK = readTopK();    // 讀目前 k

  appendMessage("user", message);
  inputBox.value = "";
  setSendingState(true);
  showTyping();

  try{
    const res = await fetch("/ask", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ message, cid: activeCid, reqId: myReq, k: topK })
    });
    const data = await res.json();
    console.log("pdf_links from backend =", data.pdf_links);
    if (myReq !== lastReqId || !data || (data.cid && data.cid !== activeCid)) {
      removeTyping();
      return;
    }
    removeTyping();
    appendMessage("assistant", data.response || "⚠️ 回答失敗", {
      bm25_titles: data.bm25_titles,
      refined_context: data.refined_context,
      k: data.k ?? topK,
      pdf_links: data.pdf_links || []
    });

    refreshConvList();
  }catch(e){
    removeTyping();
    appendMessage("assistant", "⚠️ 連線失敗");
  }finally{
    inflight = false;
    setSendingState(false);
    document.getElementById("user-input").focus();
  }
}

// ---------- boot ----------
// ---------- boot (robust, guarded) ----------
window.addEventListener("DOMContentLoaded", async () => {
  // 全域 error 註冊，方便偵錯（可移除）
  window.addEventListener("error", (ev) => {
    console.error("Window error:", ev.error || ev.message, ev);
  });
  window.addEventListener("unhandledrejection", (ev) => {
    console.error("Unhandled promise rejection:", ev.reason);
  });

  try {
    // 綁定送出按鈕（若存在）
    const sendBtn = document.getElementById("send-btn");
    if (sendBtn) sendBtn.addEventListener("click", sendMessage);

    // 綁定輸入欄（Enter 送出）
    const userInput = document.getElementById("user-input");
    if (userInput) {
      userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
        if (inflight) e.stopPropagation();
      });
    }

    // 新對話按鈕
    const newChatBtn = document.getElementById("new-chat");
    if (newChatBtn) newChatBtn.addEventListener("click", newConversation);

    // Top-K 元件綁定（只有在存在時）
    const r = document.getElementById("topkRange");
    const n = document.getElementById("topkNumber");
    const i = document.getElementById("kInput");
    if (r) r.addEventListener("input", () => syncTopK("range"));
    if (n) n.addEventListener("input", () => syncTopK("number"));
    if (i) i.addEventListener("change", () => syncTopK("input"));
    syncTopK(); // 初始同步

    // Profile (avatar) 按鈕 / Modal 綁定（如果你有新增 HTML 才會綁）
    const profileBtn = document.getElementById("profile-btn");
    const profileModal = document.getElementById("profileModal");
    const profileSave = document.getElementById("profile-save");
    if (profileBtn && profileModal) {
      profileBtn.addEventListener("click", () => {
        try {
          profileModal.style.display = "flex";
          if (typeof loadProfile === "function") {
            loadProfile();
          }
        } catch (e) {
          console.error("profile open failed:", e);
        }
      });
    }
    if (profileSave) {
      profileSave.addEventListener("click", async () => {
        try {
          const payload = {
            year: document.getElementById("p_year")?.value || "",
            degree: document.getElementById("p_degree")?.value || "",
            college: document.getElementById("p_college")?.value || "",
            dept: document.getElementById("p_dept")?.value || "",
            sid: document.getElementById("p_sid")?.value || ""
          };
          await fetch("/api/profile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          // close
          if (profileModal) profileModal.style.display = "none";
          // optional: reload conversation context or UI
          console.log("Profile saved");
        } catch (err) {
          console.error("Saving profile failed:", err);
        }
      });
    }

    // 其餘啟動動作（fetch conversations），放在 try 裡面，保護任何未定義錯誤
    await refreshConvList().catch(e => {
      console.error("refreshConvList failed:", e);
    });

    if (!activeCid) {
      await newConversation().catch(e => console.error("newConversation failed:", e));
    } else {
      await openConversation(activeCid).catch(e => console.error("openConversation failed:", e));
    }

    // focus input if exists
    if (userInput) userInput.focus();

  } catch (err) {
    console.error("Boot failed:", err);
    // 移除 loading typing（保險）
    try { removeTyping(); } catch (e) {}
    setSendingState(false);
  }
});