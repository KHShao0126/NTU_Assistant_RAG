import os
import json
import re
from typing import List, Dict, Optional
from flask import session

chat_history = []

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyMuPDF (fitz) is required. Please install with: pip install pymupdf") from exc


def _ensure_deps():
    """Ensure optional dependencies are available; raise helpful errors otherwise."""
    try:
        from rank_bm25 import BM25Okapi  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("rank-bm25 is required. Please install with: pip install rank-bm25") from exc

    # jieba is optional; we fall back to regex if unavailable
    try:
        import jieba  # noqa: F401
    except Exception:
        pass


_ensure_deps()
from rank_bm25 import BM25Okapi


def default_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 scoring.

    Prefer jieba for Chinese segmentation; otherwise, use a regex that splits
    CJK characters to single-char tokens and Latin to word tokens.
    """
    try:
        import jieba
        return [token.strip() for token in jieba.cut(text) if token.strip()]
    except Exception:
        # Keep alphanumerics as words, split CJK into single characters
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)


class BM25DocumentRetriever:
    """Builds a BM25 index at the document (per-PDF) level and retrieves top documents.

    - Each PDF file becomes one document.
    - BM25 is built over tokenized full-document texts.
    - Use search() to get the most relevant documents for a query.
    """

    def __init__(
        self,
        pdf_folder: str = "./ntu_rules_pdfs",
        corpus_path: str = "bm25_docs_big.json",
        tokenizer=default_tokenize,
    ) -> None:
        self.pdf_folder = pdf_folder
        self.corpus_path = corpus_path
        self.tokenize = tokenizer

        self.documents: List[Dict[str, str]] = []  # [{doc_id, text}]
        self._bm25: Optional[BM25Okapi] = None

    def build_or_load_corpus(self) -> None:
        """Load existing corpus JSON or build from PDFs and save it."""
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            return

        documents: List[Dict[str, str]] = []
        for filename in os.listdir(self.pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(self.pdf_folder, filename)
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                # Skip unreadable PDFs but keep going
                continue
            text_fragments: List[str] = []
            for page in doc:
                try:
                    text_fragments.append(page.get_text())
                except Exception:
                    pass
            full_text = "\n".join(text_fragments).strip()
            if not full_text:
                continue
            documents.append({"doc_id": filename, "text": full_text})

        # Persist corpus
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)

        self.documents = documents

    def build_index(self) -> None:
        """Build BM25 index for the loaded corpus."""
        if not self.documents:
            self.build_or_load_corpus()
        tokenized_docs = [self.tokenize(d["text"]) for d in self.documents]
        self._bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        """Return top-k most relevant documents by BM25 score.

        Response schema per item:
        - doc_id: str (PDF filename)
        - score: float (BM25 score)
        - text: str (full document text)
        """
        if self._bm25 is None:
            self.build_index()
        tokenized_query = self.tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results: List[Dict[str, object]] = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "score": float(scores[idx]),
                "text": doc["text"],
            })
        return results

    def build_context(self, query: str, k: int = 3, max_chars_per_doc: Optional[int] = 6000) -> str:
        """Build a concatenated context from top-k documents with optional per-doc trimming.

        This is useful if you want to pass entire documents (or trimmed versions) to an LLM.
        """
        top_docs = self.search(query, k=k)
        formatted_docs: List[str] = []
        for rank, item in enumerate(top_docs, start=1):
            text = item["text"]
            if isinstance(max_chars_per_doc, int) and max_chars_per_doc > 0:
                text = text[:max_chars_per_doc]
            header = f"[Document {rank}] {item['doc_id']} (score={item['score']:.3f})"
            formatted_docs.append(f"{header}\n{text}")
        return "\n\n---\n\n".join(formatted_docs)
    

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re

# Try to ensure BM25 is available
try:
    from rank_bm25 import BM25Okapi
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rank-bm25"])  # silent install
    from rank_bm25 import BM25Okapi

# Simple tokenizer that works for Chinese and Latin text
try:
    import jieba
    def tokenize(text: str):
        return [t for t in jieba.cut(text) if t.strip()]
except Exception:
    def tokenize(text: str):
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)

MODEL = None
TOKENIZER = None

# 載入模型
model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "google/gemma-3-12b-it"

# 選擇裝置（CUDA，Apple MPS，CPU）
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 模型快取
if MODEL is None or TOKENIZER is None:
    print("Loading model and tokenizer (this may take a few minutes the first time)...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
    ).to(device)
else:
    print("Using cached model and tokenizer.")

tokenizer = TOKENIZER
model = MODEL

# 建立推理管線（Optional）
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False)



def _extract_titles_from_context(context: str):
    # 例："[Document 1] 某某規章.pdf (score=0.812)"
    import re
    return re.findall(r"\[Document\s+\d+\]\s*(.*?)\s*\(score=", context or "")



def generate_prompt(user_input, context, conversation_history):
    if not isinstance(conversation_history, list):
        conversation_history = []
    cleaned_history = []
    for turn in conversation_history:
        if isinstance(turn, dict) and "user" in turn and "assistant" in turn:
            cleaned_history.append({"user": str(turn["user"]), "assistant": str(turn["assistant"])})
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            cleaned_history.append({"user": str(turn[0]), "assistant": str(turn[1])})
        else:
            continue
    #titles = _extract_titles_from_context(context)
    print(context)

    print("【學生問題】", user_input)
    history_str = ""
    for turn in cleaned_history:
        history_str += f"使用者：{turn['user']}\n法規助理：{turn['assistant']}\n"
    print("【對話歷史】", history_str if history_str else "無")
    return f"""你是一位台大學生的法規助理，請根據以下資料回答學生的問題。

[對話歷史]
{history_str}

[法規資料]
{context}

[學生問題]
{user_input}

請給出準確、清楚的回覆，若資料不足，請說明還需要哪些學生資訊。回答要簡潔，若法規資料中有跟學生問題無關的請忽略，一定不用多餘的說明。"""


def call_qwen(prompt):
    profile = session.get("profile", {})

    year = profile.get("year", "（未設定入學年份）")
    degree = profile.get("degree", "（未設定學位）")
    college = profile.get("college", "（未設定學院）")
    dept = profile.get("dept", "（未設定學系）")
    sid = profile.get("sid", "（未設定學號）")

    system_prompt = f"""
    你是一位台大學生的法規助理。
    學生背景如下：
    - 入學年份：{year}
    - 學位：{degree}
    - 學院：{college}
    - 學系：{dept}
    - 學號：{sid}

    請盡可能根據學生的身分給出更貼近情況的建議。
    """

    print(system_prompt)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

    #  1. 產生聊天格式 prompt（純文字）
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    #  2. 用 tokenizer 編碼成 input_ids + attention_mask
    encoded = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    #  3. 模型產生
    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=512,
        do_sample=False
    )

    #  4. 解碼結果
    output_ids = outputs[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    #  5. 移除 prompt 前綴（可選）
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response


#  測試範例（使用 BM25 擷取 top-k document 作為 context）
#question = "我現在學士班大三，沒有輔系，已經修了83學分，我還差多少才能畢業？"
#question = "一學期想要修超過25學分的資格是什麼？"
#retriever = BM25DocumentRetriever(pdf_folder="./ntu_rules_pdfs", corpus_path="bm25_docs_big.json")
#retriever.build_or_load_corpus()
#retriever.build_index()
#results = retriever.search(question, k=5)
#print("Top-5 documents:")
#for i, r in enumerate(results, 1):
    #preview = r["text"][:120].replace("\n", " ")
    #print(f"{i}. {r['doc_id']}  score={r['score']:.3f}  preview={preview}{'...' if len(r['text'])>120 else ''}")

#print("\nContext to feed the LLM (trimmed per doc):")
#context = retriever.build_context(question, k=5, max_chars_per_doc=2000)
#print(context[:2000])


# --------------- 新增：LLM Reranking 條文選段階段 ---------------

def llm_rerank_relevant_passages(query: str, bm25_context: str) -> str:
    print(bm25_context)
    """使用同一個模型，根據問題在 BM25 context 中選出最相關條文。"""
    rerank_prompt = f"""你是一位負責資料擷取的助理。你負責從文件中保留與學生問題直接相關的條文或段落。忽略與問題無關的內容。不要加任何解釋或分析。

[學生問題]
{query}

[文件]
{bm25_context}\n\n

請以 
根據[Document X] pdf檔名 相關內容 
的格式，輸出與學生問題最相關的條文或段落。
"""
    print("\n====== LLM 條文選段階段 ======")
    selected_text = call_qwen(rerank_prompt)
    print(selected_text)
    return selected_text

#refined_context = llm_rerank_relevant_passages(question, context)
#prompt = generate_prompt(question, refined_context, chat_history)
#answer = call_qwen(prompt)

# 更新歷史
#chat_history.append({"user": question, "assistant": answer})

#print("🤖 回答：")
#print(answer)
#print(refined_context)




# In[10]:


#  測試範例
#question = "幫我規劃剩下的學分該如何修完"
#prompt = generate_prompt(question, context, chat_history)
#answer = call_qwen(prompt)

#print("🤖 回答：")
#print(answer)"""


#114
