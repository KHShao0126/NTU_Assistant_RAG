#!/usr/bin/env python
# coding: utf-8

# In[2]:




# In[7]:


import os
import json
import re
from typing import List, Dict, Optional

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
        pdf_folder: str = "./台大資工相關規範",
        corpus_path: str = "bm25_docs.json",
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

    def build_context(self, query: str, k: int = 5, max_chars_per_doc: Optional[int] = 6000) -> str:
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








# In[9]:


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

# 載入較小的 Qwen 模型（1.8B chat 版較容易在 16GB RAM 上運行）
model_id = "Qwen/Qwen1.5-7B-Chat"

# 選擇裝置（優先 CUDA，其次 Apple MPS，最後 CPU）
if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# 模型快取
if MODEL is None or TOKENIZER is None:
    print("Loading Qwen model and tokenizer (this may take a few minutes the first time)...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
    ).to(device)
else:
    print("Using cached Qwen model and tokenizer.")

tokenizer = TOKENIZER
model = MODEL

# 建立推理管線（可選，不一定要使用）
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=False)







def generate_prompt(question, context):
    return f"""你是一位台大資工系的法規助理，請根據以下資料回答學生的問題。

[法規資料]
{context}

[學生問題]
{question}

請給出準確、清楚的回覆，若資料不足，請說明還需要哪些學生資訊。"""


def call_qwen(prompt):
    messages = [
        {"role": "system", "content": "你是台大資工系的法規助理，請根據資料回答問題。 提出問題的都是台大資工的學生"},
        {"role": "user", "content": prompt}
    ]

    # ✅ 1. 產生聊天格式 prompt（純文字）
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ✅ 2. 用 tokenizer 編碼成 input_ids + attention_mask
    encoded = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # ✅ 3. 模型產生
    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=512,
        do_sample=False
    )

    # ✅ 4. 解碼結果
    output_ids = outputs[0]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # ✅ 5. 移除 prompt 前綴（可選）
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()

    return response

# ✅ 測試範例（使用 BM25 擷取 top-k chunk 作為 context）
question = "我現在學士班大三，沒有輔系，已經修了83學分，我還差多少才能畢業？"
retriever = BM25DocumentRetriever(pdf_folder="./台大資工相關規範", corpus_path="bm25_docs.json")
retriever.build_or_load_corpus()
retriever.build_index()
results = retriever.search(question, k=5)
print("Top-5 documents:")
for i, r in enumerate(results, 1):
    preview = r["text"][:120].replace("\n", " ")
    print(f"{i}. {r['doc_id']}  score={r['score']:.3f}  preview={preview}{'...' if len(r['text'])>120 else ''}")

print("\nContext to feed the LLM (trimmed per doc):")
context = retriever.build_context(question, k=5, max_chars_per_doc=2000)
print(context[:2000])
prompt = generate_prompt(question, context)
answer = call_qwen(prompt)

print("🤖 回答：")
print(answer)


# In[10]:


# ✅ 測試範例
question = "我需要修哪幾類的通識才可以畢業"
prompt = generate_prompt(question, context)
answer = call_qwen(prompt)

print("🤖 回答：")
print(answer)

