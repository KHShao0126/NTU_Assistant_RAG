import os
import json
import re  # <-- Used extensively
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

# --- Multi-Layer Embedding Retriever (Flexible V2) ---

class EmbeddingDocumentRetriever:
    """
    Builds a flexible multi-layer embedding index based on the document's
    hierarchical structure. It tries multiple common patterns.
    
    Each chunk is embedded with its parent's text prepended for context.
    
    Uses BAAI/bge-m3, a long-context model, to handle combined parent/child text.
    """

    def __init__(
        self,
        pdf_folder: str = "../ntu_rules_pdfs",
        # --- CHANGED: New paths for V2 index with BGE-M3 ---
        corpus_path: str = "multilayer_corpus_bge-m3.json",
        index_path: str = "multilayer_bge-m3.index",
        # --- CHANGED: Using BGE-M3 for its long context (8192 tokens) ---
        model_name: str = 'BAAI/bge-m3',
    ) -> None:
        self.pdf_folder = pdf_folder
        self.corpus_path = corpus_path
        self.index_path = index_path
        
        self.device = self._get_device()
        print(f"Using device for embeddings: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        # --- ADDED: Set max sequence length for BGE-M3 ---
        self.model.max_seq_length = 8192
        
        self.documents: List[Dict[str, object]] = []  # [{doc_id, layer, ...}]
        self.index: Optional[faiss.Index] = None

        # --- ADDED: More flexible regex patterns ---
        CHINESE_NUMERALS = r'[一二三四五六七八九十百]'
        
        # L1 Patterns (Articles or main sections)
        self.l1_article_pattern = re.compile(
            rf'^(第{CHINESE_NUMERALS}+條)', re.MULTILINE
        )
        self.l1_numeral_pattern = re.compile(
            rf'^( ?{CHINESE_NUMERALS}+、)', re.MULTILINE
        )
        
        # L2 Patterns (Clauses or sub-sections)
        self.l2_paren_pattern = re.compile(
            rf'^\s*(\({CHINESE_NUMERALS}+\))', re.MULTILINE
        )
        # L2 can also be numerals (e.g., in Honors Program doc)
        self.l2_numeral_pattern = re.compile(
            rf'^( ?{CHINESE_NUMERALS}+、)', re.MULTILINE
        )
        # -------------------------------------------

    def _get_device(self) -> str:
        """Selects the best available device for torch."""
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # --- HEAVILY MODIFIED METHOD ---
    def build_or_load_corpus_and_index(self) -> None:
        """
        Loads corpus and FAISS index from disk if they exist.
        Otherwise, parses all PDFs, builds a flexible multi-layer corpus,
        creates embeddings, and saves both to disk.
        """
        if os.path.exists(self.corpus_path) and os.path.exists(self.index_path):
            print("Loading existing multi-layer corpus and FAISS index from disk...")
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                self.index = faiss.read_index(self.index_path)
                print("Loading complete.")
                return
            except Exception as e:
                print(f"Error loading files (will rebuild): {e}")

        # --- Build from Scratch ---
        print("Building new multi-layer corpus and FAISS index...")
        if not os.path.isdir(self.pdf_folder):
            print(f"Error: PDF folder not found at {self.pdf_folder}")
            raise FileNotFoundError(self.pdf_folder)
            
        print(f"Parsing PDFs from: {self.pdf_folder}")
        
        documents: List[Dict[str, object]] = []
        
        for filename in os.listdir(self.pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue
            
            pdf_path = os.path.join(self.pdf_folder, filename)
            
            try:
                doc = fitz.open(pdf_path)
            except Exception:
                print(f"Warning: Could not open {filename}. Skipping.")
                continue
                
            text_fragments: List[str] = []
            for page in doc:
                try:
                    # Use "sort=True" to maintain logical text order
                    text_fragments.append(page.get_text("text", sort=True))
                except Exception:
                    pass
            full_text = "\n".join(text_fragments).strip()
            
            if not full_text:
                print(f"Warning: No text extracted from {filename}. Skipping.")
                continue

            # --- Start Multi-Layer Parsing (Flexible) ---
            
            # Layer 0 (Document)
            doc_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
            documents.append({
                "doc_id": filename,
                "layer": 0,
                "title": doc_title,
                "text_to_embed": f"{doc_title}\n\n{full_text}",
                "original_text_for_display": full_text
            })
            
            # --- Try different L1 parsing strategies ---
            article_parts = self.l1_article_pattern.split(full_text)
            
            if len(article_parts) < 3:
                # If '第X條' fails, try '一、', '二、'
                article_parts = self.l1_numeral_pattern.split(full_text)
            
            if len(article_parts) < 3:
                # If both fail, treat as unstructured
                print(f"Warning: No L1 structure (第X條 or 一、) found in {filename}. Only L0 (full doc) chunk was created.")
                continue
            # -------------------------------------------

            # Iterate through [Article Title, Article Body] pairs
            for i in range(1, len(article_parts), 2):
                article_title_text = article_parts[i].strip()
                article_body_text = article_parts[i+1].strip()
                full_article_text = f"{article_title_text}\n{article_body_text}"
                
                # Layer 1 (Article)
                parent_text_L0 = doc_title
                text_to_embed_L1 = f"{parent_text_L0}\n\n{full_article_text}"
                documents.append({
                    "doc_id": filename,
                    "layer": 1,
                    "article": article_title_text,
                    "text_to_embed": text_to_embed_L1,
                    "original_text_for_display": full_article_text
                })
                
                # --- Try different L2 parsing strategies ---
                clause_parts = self.l2_paren_pattern.split(article_body_text)
                
                if len(clause_parts) < 3:
                    # If '(一)' fails, try '一、' (for docs like Honors Program)
                    clause_parts = self.l2_numeral_pattern.split(article_body_text)
                
                if len(clause_parts) <= 1: # No L2 clauses found
                    continue 
                # ------------------------------------------
                    
                # The 'caput' (lead-in text) is the part before the first clause
                caput_text = clause_parts[0].strip()
                parent_text_L1 = f"{article_title_text}\n{caput_text}"

                # Iterate through [Clause Title, Clause Body] pairs
                for j in range(1, len(clause_parts), 2):
                    clause_title_text = clause_parts[j].strip()
                    # Clean up body text, remove newlines
                    clause_body_text = ' '.join(clause_parts[j+1].strip().splitlines())
                    full_clause_text = f"{clause_title_text} {clause_body_text}"
                    
                    # Layer 2 (Clause)
                    text_to_embed_L2 = f"{parent_text_L1}\n\n{full_clause_text}"
                    documents.append({
                        "doc_id": filename,
                        "layer": 2,
                        "article": article_title_text,
                        "clause": clause_title_text,
                        "text_to_embed": text_to_embed_L2,
                        "original_text_for_display": full_clause_text
                    })

        if not documents:
            print(f"Error: No documents were successfully parsed from {self.pdf_folder}.")
            raise ValueError("No documents found to index.")

        # Persist corpus
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        self.documents = documents
        print(f"Parsed and saved {len(self.documents)} multi-layer chunks to {self.corpus_path}")

        # --- Create and save FAISS index ---
        print("Creating embeddings for multi-layer chunks (this may take a while)...")
        doc_texts_to_embed = [d['text_to_embed'] for d in self.documents]
        
        embeddings = self.model.encode(
            doc_texts_to_embed, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            # --- ADDED: Normalize embeddings, recommended for BGE ---
            normalize_embeddings=True
        )
        
        embeddings_np = embeddings.astype('float32')
        # --- REMOVED: No need to normalize manually ---
        # faiss.normalize_L2(embeddings_np)
        
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # IP = Inner Product (same as Cosine Sim on normalized)
        self.index.add(embeddings_np)
        
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")
        print("Building complete.")
    # --- END OF MODIFIED METHOD ---

    def search(self, query: str, k: int = 5) -> List[Dict[str, object]]:
        """
        Return top-k most relevant chunks by semantic similarity.
        """
        if self.index is None:
            print("Index not built. Building/loading now.")
            self.build_or_load_corpus_and_index()
            if self.index is None:
                raise RuntimeError("Failed to build or load index.")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        k = min(k, len(self.documents))

        # --- UPDATED: Normalize query embedding to match ---
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_embedding_np = query_embedding.astype('float32')
        # --- REMOVED: No need to normalize manually ---
        # faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding_np, k)
        
        results: List[Dict[str, object]] = []
        for i in range(k):
            if i >= len(indices[0]):
                break
            idx = indices[0][i]
            if idx < 0 or idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            
            # Build a descriptive ID for clarity
            display_id = doc.get("title", doc.get("doc_id", "Unknown"))
            if "article" in doc and doc["article"] != "Full Document":
                display_id = f"{display_id} - {doc['article']}"
            if "clause" in doc:
                display_id = f"{display_id} {doc['clause']}"
            
            results.append({
                "doc_id": display_id,
                "score": float(distances[0][i]),
                "text": doc["original_text_for_display"], # Return clean text
            })
        return results

    def build_context(self, query: str, k: int = 3, max_chars_per_doc: Optional[int] = 6000) -> str:
        """
        Build a concatenated context from top-k chunks.
        """
        top_docs = self.search(query, k=k)
        formatted_docs: List[str] = []
        for rank, item in enumerate(top_docs, start=1):
            text = item["text"]
            if isinstance(max_chars_per_doc, int) and max_chars_per_doc > 0:
                text = text[:max_chars_per_doc]
            header = f"[Chunk {rank}] {item['doc_id']} (score={item['score']:.3f})"
            formatted_docs.append(f"{header}\n{text}")
        
        if not formatted_docs:
            return "[No relevant documents found]"
            
        return "\n\n---\n\n".join(formatted_docs)
    
# --- LLM Pipeline ---
# (Using the cleaner 'pipeline' implementation from your previous script)

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
            cleaned_heading = {"user": str(turn["user"]), "assistant": str(turn["assistant"])}
            cleaned_history.append(cleaned_heading)
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            cleaned_heading = {"user": str(turn[0]), "assistant": str(turn[1])}
            cleaned_history.append(cleaned_heading)
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
    
    final_prompt = f"""你是一位國立台灣大學的法規助理，請根據以下資料回答學生的問題。

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
    response_data = llm(messages, pad_token_id=TOKENIZER.eos_token_id)
    
    # Extract the assistant's reply
    response = response_data[0]['generated_text'][-1]['content']
    
    end_time = time.time()
    print(f"LLM generation took {end_time - start_time:.2f} seconds.")

    return response

# --------------- LLM Reranking 條文選段階段 ---------------

def llm_rerank_relevant_passages(query: str, retrieved_context: str) -> str:
    """使用同一個模型，根據問題在 retrieved_context 中選出最相關條文。"""
    
    # --- BUG FIX: Was 'retrieved_T', changed to 'retrieved_context' ---
    rerank_prompt = f"""你是一位負責資料擷取的助理。你的回答只要保留與學生問題直接相關的條文或段落。忽略與問題無關的內容。不要加任何解釋或分析。

[學生問題]
{query}

[文件]
{retrieved_context}
"""
    # -------------------------------------------------------------
    
    print("\n====== LLM Reranking Stage ======")
    selected_text = call_llm(rerank_prompt)
    print(selected_text)
    return selected_text

# --- Main Execution (Interactive Loop) ---

if __name__ == "__main__":
    
    chat_history = [] 

    try:
        # --- One-Time Setup ---
        print("--- Initializing Multi-Layer Embedding Retriever (BGE-M3) ---")
        # --- UPDATED: Using new paths for BGE-M3 version ---
        retriever = EmbeddingDocumentRetriever(
            pdf_folder="../ntu_rules_pdfs", 
            corpus_path="multilayer_corpus_bge-m3.json",
            index_path="multilayer_bge-m3.index"
        )
        
        retriever.build_or_load_corpus_and_index()
        
        print("\n--- Initializing LLM ---")
        load_llm_model(model_id="meta-llama/Llama-3.1-8B-Instruct")
        
        print("\n---  法規助理 (Multi-Layer BGE-M3) Ready ---")
        print("您好！我是台大的法規助理。")
        print("您可以問我有關規定的問題。")
        print("Commands:")
        print("  - Type 'clear' or 'clear history' to clear conversation history")
        print("  - Type 'exit', 'quit', or 'q' to end chat")
        print("-" * 30)

        # --- Interactive Loop ---
        while True:
            # --- CLEANUP: Removed redundant commands print ---
            question = input("學生: ")
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("法規助理: 掰掰！")
                break
            
            if question.lower() in ['clear', 'clear history', 'reset']:
                chat_history = []
                print("法規助理: 對話歷史已清空。")
                continue
                
            if not question.strip():
                continue

            # --- RAG Pipeline ---
            print(f"\n--- Stage 1: Semantic Search (Top-3 Chunks) ---") # <-- Changed to Top-3
            start_time = time.time()
            results = retriever.search(question, k=3) # <-- Use k=3
            end_time = time.time()
            print(f"Search took {end_time - start_time:.4f} seconds.")

            print("\nTop-3 relevant chunks:") # <-- Changed to Top-3
            for i, r in enumerate(results, 1):
                # 'doc_id' is now descriptive
                print(f"  {i}. {r['doc_id']} (Score: {r['score']:.3f})")

            # Context for reranking
            context_for_generation = retriever.build_context(question, k=3, max_chars_per_doc=2000) # <-- Use k=3
            
            # --- Stage 2: LLM Reranking (SKIPPED) ---
            # refined_context = llm_rerank_relevant_passages(question, context_for_reranking)

            # Stage 3: Final Answer Generation
            # --- Use the direct context from the retriever ---
            prompt = generate_prompt(question, context_for_generation, chat_history)
            answer = call_llm(prompt)

            # Update history and print answer
            chat_history.append({"user": question, "assistant": answer})
            print("\n" + "=" * 20 + " 法規助理 " + "=" * 20)
            print(answer)
            print("=" * (44 + len("法規助理")))
            print("\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n法規助理: 收到中斷訊號，掰掰！")



