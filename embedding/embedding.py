import os
import json
import re
from typing import List, Dict, Optional
import time

# --- Dependency Checks ---
try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyMuPDF (fitz) is required. Please install with: pip install pymupdf") from exc

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("sentence-transformers is required. Please install with: pip install sentence-transformers") from exc

try:
    import faiss
except Exception as exc:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required. Please install with: pip install faiss-cpu") from exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise RuntimeError("numpy is required. Please install with: pip install numpy") from exc

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("transformers and torch are required. Please install: pip install transformers torch") from exc

# --- Embedding Retriever (Document-Level) ---

class EmbeddingDocumentRetriever:
    """
    Builds an embedding index at the document (per-PDF) level.
    - Each PDF file becomes one document.
    - Uses BAAI/bge-m3, a long-context model (8192 tokens).
    """

    def __init__(
        self,
        pdf_folder: str = "../ntu_rules_pdfs",
        corpus_path: str = "doc_corpus_bge-m3.json",  
        index_path: str = "doc_index_bge-m3.index",    
        model_name: str = 'BAAI/bge-m3',             # <-- LONG CONTEXT MODEL
    ) -> None:
        self.pdf_folder = pdf_folder
        self.corpus_path = corpus_path
        self.index_path = index_path
        
        self.device = self._get_device()
        print(f"Using device for embeddings: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        # bge-m3 has a max sequence length of 8192
        self.model.max_seq_length = 8192
        
        self.documents: List[Dict[str, str]] = []  # [{doc_id, text}]
        self.index: Optional[faiss.Index] = None

    def _get_device(self) -> str:
        """Selects the best available device for torch."""
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def build_or_load_corpus_and_index(self) -> None:
        """
        Loads corpus and FAISS index from disk if they exist.
        Otherwise, parses all PDFs, builds corpus, creates embeddings,
        and saves both to disk.
        """
        if os.path.exists(self.corpus_path) and os.path.exists(self.index_path):
            print("Loading existing corpus and FAISS index from disk...")
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self.index = faiss.read_index(self.index_path)
                print("Loading complete.")
                return
            except Exception as e:
                print(f"Error loading files (will rebuild): {e}")

        # --- Build from Scratch ---
        print("Building new corpus and FAISS index...")
        if not os.path.isdir(self.pdf_folder):
            print(f"Error: PDF folder not found at {self.pdf_folder}")
            print("Please check the 'pdf_folder' path in EmbeddingDocumentRetriever.")
            raise FileNotFoundError(self.pdf_folder)
            
        print(f"Parsing PDFs from: {self.pdf_folder}")
        
        documents: List[Dict[str, str]] = []
        for filename in os.listdir(self.pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(self.pdf_folder, filename)
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                print(f"Warning: Could not open {filename}. Skipping.")
                continue # Skip unreadable
                
            text_fragments: List[str] = []
            for page in doc:
                try:
                    text_fragments.append(page.get_text())
                except Exception:
                    pass
            full_text = "\n".join(text_fragments).strip()
            
            if not full_text:
                print(f"Warning: No text extracted from {filename}. Skipping.")
                continue
            documents.append({"doc_id": filename, "text": full_text})

        if not documents:
            print(f"Error: No documents were successfully parsed from {self.pdf_folder}.")
            print("Please ensure the folder is not empty and PDFs are readable.")
            raise ValueError("No documents found to index.")

        # Persist corpus text
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)
        
        self.documents = documents
        print(f"Parsed and saved {len(self.documents)} documents to {self.corpus_path}")

        # --- Create and save FAISS index ---
        print("Creating embeddings (this may take a while)...")
        doc_texts = [d['text'] for d in self.documents]
        
        embeddings = self.model.encode(
            doc_texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True # Recommended for BGE models
        )
        
        embeddings_np = embeddings.astype('float32')
        
        # We use IndexFlatIP because embeddings are normalized
        # (Cosine Similarity is Inner Product on normalized vectors)
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_np)
        
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")
        print("Building complete.")

    def search(self, query: str, k: int = 3) -> List[Dict[str, object]]:
        """
        Return top-k most relevant documents by semantic similarity.
        """
        if self.index is None:
            print("Index not built. Building/loading now.")
            self.build_or_load_corpus_and_index()
            if self.index is None:
                raise RuntimeError("Failed to build or load index.")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k = min(k, len(self.documents))

        # Embed the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True # Match document normalization
        )
        
        query_embedding_np = query_embedding.astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding_np, k)
        
        results: List[Dict[str, object]] = []
        for i in range(k):
            if i >= len(indices[0]):
                break
                
            idx = indices[0][i]
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            results.append({
                "doc_id": doc["doc_id"],
                "score": float(distances[0][i]), # Cosine similarity score
                "text": doc["text"],
            })
        return results

    def build_context(self, query: str, k: int = 3, max_chars_per_doc: Optional[int] = 6000) -> str:
        """
        Build a concatenated context from top-k documents.
        """
        top_docs = self.search(query, k=k)
        formatted_docs: List[str] = []
        for rank, item in enumerate(top_docs, start=1):
            text = item["text"]
            if isinstance(max_chars_per_doc, int) and max_chars_per_doc > 0:
                text = text[:max_chars_per_doc]
            header = f"[Document {rank}] {item['doc_id']} (score={item['score']:.3f})"
            formatted_docs.append(f"{header}\n{text}")
        
        if not formatted_docs:
            return "[No relevant documents found]"
            
        return "\n\n---\n\n".join(formatted_docs)
    
# --- LLM Pipeline ---

MODEL = None
TOKENIZER = None
llm = None # Define llm globally

def load_llm_model(model_id: str) -> None:
    """Loads the LLM and Tokenizer into global variables."""
    global MODEL, TOKENIZER, llm
    
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device for LLM: {device}")

    print("Loading model and tokenizer (this may take a few minutes the first time)...")
    TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32
    ).to(device)
    
    # Create inference pipeline
    llm = pipeline(
        "text-generation", 
        model=MODEL, 
        tokenizer=TOKENIZER, 
        max_new_tokens=512, 
        do_sample=False,
        device=MODEL.device
    )
    print("LLM loaded successfully.")


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
    
    print("\n--- Generating Final Prompt ---")
    print(f"【學生問題】 {user_input}")
    
    history_str = ""
    for turn in cleaned_history:
        history_str += f"使用者：{turn['user']}\n法規助理：{turn['assistant']}\n"
    
    if history_str:
        print(f"【對話歷史】\n{history_str.strip()}")
    else:
        print("【對話歷史】 無")
    
    print(f"【法規資料 (Refined Context)】\n{context[:1000]}...") # Truncate for display
    
    final_prompt = f"""你是一位台大資工系的法規助理，請根據以下資料回答學生的問題。

[對話歷史]
{history_str}

[法規資料]
{context}

[學生問題]
{user_input}

請給出準確、清楚的回覆，若資料不足，請說明還需要哪些學生資訊。回答要簡潔，若法規資料中有跟學生問題無關的請忽略，一定不用多餘的說明。"""
    return final_prompt


def call_llm(prompt):
    """
    Generates a response using the Llama LLM model.
    """
    global llm
    if llm is None:
        raise RuntimeError("LLM pipeline is not initialized. Call load_llm_model() first.")
        
    messages = [
        {"role": "system", "content": "你是國立台灣大學的法規助理，請根據資料回答問題。 提出問題的都是國立台灣大學的學生"},
        {"role": "user", "content": prompt}
    ]

    print("Generating LLM response...")
    start_time = time.time()
    
    # Use the pipeline directly
    # The pipeline handles chat templating automatically
    response_data = llm(messages, pad_token_id=TOKENIZER.eos_token_id)
    
    # Extract the assistant's reply from the full chat history
    response = response_data[0]['generated_text'][-1]['content']
    
    end_time = time.time()
    print(f"LLM generation took {end_time - start_time:.2f} seconds.")

    return response

# --------------- LLM Reranking 條文選段階段 ---------------

def llm_rerank_relevant_passages(query: str, retrieved_context: str) -> str:
    """使用同一個模型，根據問題在 retrieved_context 中選出最相關條文。"""
    rerank_prompt = f"""你是一位負責資料擷取的助理。你的回答只要保留與學生問題直接相關的條文或段落。忽略與問題無關的內容。不要加任何解釋或分析。

[學生問題]
{query}

[文件]
{retrieved_context}
"""
    print("\n====== LLM Reranking Stage ======")
    selected_text = call_llm(rerank_prompt)
    print(selected_text)
    return selected_text

# --- Main Execution (Interactive Loop) ---

if __name__ == "__main__":
    
    chat_history = [] # Initialize chat history

    try:
        # --- One-Time Setup ---
        print("--- Initializing Document-Level Embedding Retriever (BGE-M3) ---")
        retriever = EmbeddingDocumentRetriever(
            pdf_folder="../ntu_rules_pdfs", 
            corpus_path="doc_corpus_bge-m3.json", # <-- New path
            index_path="doc_index_bge-m3.index"  # <-- New path
        )
        
        retriever.build_or_load_corpus_and_index()
        
        print("\n--- Initializing LLM ---")
        load_llm_model(model_id="meta-llama/Llama-3.1-8B-Instruct")
        
        print("\n--- [LLM Assistant] 法規助理 Ready ---")
        print("您好！我是台大的法規助理。")
        print("您可以問我有關規定的問題。")
        print("Commands:")
        print("  - Type 'clear' or 'clear history' to clear conversation history")
        print("  - Type 'exit', 'quit', or 'q' to end chat")
        print("-" * 30)

        # --- Interactive Loop ---
        while True:
            print("Commands:")
            print("  - Type 'clear' or 'clear history' to clear conversation history")
            print("  - Type 'exit', 'quit', or 'q' to end chat")
            print("-" * 30)
            question = input("學生: ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("[LLM Assistant] 法規助理: 掰掰！")
                break
            
            if question.lower() in ['clear', 'clear history', 'reset']:
                chat_history = []
                print("[LLM Assistant] 法規助理: 對話歷史已清空。")
                continue
                
            if not question.strip():
                continue

            # --- RAG Pipeline ---
            print(f"\n--- Stage 1: Semantic Search (Top-5 Documents) ---")
            start_time = time.time()
            results = retriever.search(question, k=3)
            end_time = time.time()
            print(f"Search took {end_time - start_time:.4f} seconds.")

            print("\nTop-3 relevant documents:")
            for i, r in enumerate(results, 1):
                preview = r["text"][:120].replace("\n", " ")
                print(f"  {i}. {r['doc_id']} (Score: {r['score']:.3f})")

            # Context for reranking (full text of top 3 docs, truncated)
            context_for_reranking = retriever.build_context(question, k=3, max_chars_per_doc=2000)
            
            # Stage 2: LLM Reranking
            refined_context = llm_rerank_relevant_passages(question, context_for_reranking)

            # Stage 3: Final Answer Generation
            prompt = generate_prompt(question, refined_context, chat_history)
            answer = call_llm(prompt)

            # Update history and print answer
            chat_history.append({"user": question, "assistant": answer})
            print("\n" + "=" * 20 + " [LLM Assistant] 法規助理 " + "=" * 20)
            print(answer)
            print("=" * (44 + len("[LLM Assistant] 法規助理")))
            print("\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n[LLM Assistant] 法規助理: 收到中斷訊號，掰掰！")
