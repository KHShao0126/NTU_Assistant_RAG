"""
Evaluation script for NTU RAG System.

For each query in query_set.json:
- Retrieves top-k relevant documents using the original retriever classes
- Compares retrieved documents against ground truth references
- Calculates Precision and Recall metrics
- Saves results to a JSON file

Usage:
    python evaluation.py
"""

import os
import sys
import json
import time
import argparse
import importlib.util
from typing import List, Dict, Tuple
from pathlib import Path

# --- Import original retriever classes ---
embedding_dir = os.path.join(os.path.dirname(__file__), 'embedding')
sys.path.insert(0, embedding_dir)

# Import directly from the module files
import importlib.util

# Load EmbeddingDocumentRetriever from embedding.py
spec_doc = importlib.util.spec_from_file_location("embedding_module", os.path.join(embedding_dir, "embedding.py"))
embedding_module = importlib.util.module_from_spec(spec_doc)
spec_doc.loader.exec_module(embedding_module)
EmbeddingDocumentRetriever = embedding_module.EmbeddingDocumentRetriever

# Load MultiLayerEmbeddingRetriever from multilayer_embedding.py
spec_ml = importlib.util.spec_from_file_location("multilayer_module", os.path.join(embedding_dir, "multilayer_embedding.py"))
multilayer_module = importlib.util.module_from_spec(spec_ml)
spec_ml.loader.exec_module(multilayer_module)
MultiLayerEmbeddingRetriever = multilayer_module.EmbeddingDocumentRetriever


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def normalize_filename(filename: str) -> str:
    """
    Normalize filename for comparison.
    - Extracts just the filename before .pdf
    - Case-insensitive
    - Strips whitespace
    
    Examples:
        "選課辦法.pdf" -> "選課辦法"
        "選課辦法.pdf - 第十二條" -> "選課辦法"
    """
    # Extract filename before .pdf if it exists
    if ".pdf" in filename:
        filename = filename.split(".pdf")[0] + ".pdf"
    
    # Remove .pdf extension
    filename = filename.lower().replace(".pdf", "").strip()
    
    return filename


def calculate_precision_recall(
    retrieved_files: List[str],
    ground_truth_files: List[str]
) -> Tuple[float, float]:
    """
    Calculate Precision and Recall.
    
    Args:
        retrieved_files: List of retrieved file names
        ground_truth_files: List of ground truth reference file names
    
    Returns:
        (precision, recall)
    """
    # Normalize for comparison
    retrieved_set = {normalize_filename(f) for f in retrieved_files}
    ground_truth_set = {normalize_filename(f) for f in ground_truth_files}
    
    # Intersection and calculations
    intersection = retrieved_set & ground_truth_set
    
    precision = len(intersection) / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    recall = len(intersection) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
    
    return precision, recall


def calculate_mean_reciprocal_rank(
    retrieved_files: List[str],
    ground_truth_files: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    MRR is the reciprocal of the rank of the first relevant document.
    
    Args:
        retrieved_files: List of retrieved file names (in order)
        ground_truth_files: List of ground truth reference file names
    
    Returns:
        MRR score (0.0 to 1.0)
    """
    ground_truth_set = {normalize_filename(f) for f in ground_truth_files}
    
    for rank, retrieved_file in enumerate(retrieved_files, 1):
        if normalize_filename(retrieved_file) in ground_truth_set:
            return 1.0 / rank
    
    return 0.0  # No relevant document found


def calculate_average_precision(
    retrieved_files: List[str],
    ground_truth_files: List[str]
) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = (sum of precisions at each relevant position) / (number of relevant documents)
    
    Formula:
    AP = Σ(P(k) × rel(k)) / |relevant documents|
    where P(k) is precision at rank k, and rel(k) = 1 if item k is relevant, 0 otherwise
    
    Args:
        retrieved_files: List of retrieved file names (in order)
        ground_truth_files: List of ground truth reference file names
    
    Returns:
        AP score (0.0 to 1.0)
    """
    ground_truth_set = {normalize_filename(f) for f in ground_truth_files}
    
    if len(ground_truth_set) == 0:
        return 0.0
    
    precisions_at_relevant = []
    num_relevant_found = 0
    already_counted = set()  # Track which documents we've already counted
    
    # Normalize retrieved files and track unique ones
    normalized_retrieved = [normalize_filename(f) for f in retrieved_files]
    
    for rank, norm_retrieved_file in enumerate(normalized_retrieved, 1):
        # Only count each relevant document once (at its first occurrence)
        if (norm_retrieved_file in ground_truth_set and 
            norm_retrieved_file not in already_counted):
            num_relevant_found += 1
            already_counted.add(norm_retrieved_file)
            precision_at_k = num_relevant_found / rank
            precisions_at_relevant.append(precision_at_k)
    
    # If no relevant documents found, AP = 0
    if len(precisions_at_relevant) == 0:
        return 0.0
    
    # AP is the sum of precisions divided by total number of relevant documents
    # This ensures AP is always in [0, 1]
    ap = sum(precisions_at_relevant) / len(ground_truth_set)
    
    return ap


def evaluate_retriever(
    retriever,
    query_set: List[Dict],
    k: int = 5,
    retriever_name: str = "Unnamed"
) -> Tuple[List[Dict], float, float, float, float]:
    """
    Evaluate a retriever on the query set.
    
    Args:
        retriever: Retriever object with search() method
        query_set: List of queries with ground truth
        k: Number of top results to retrieve
        retriever_name: Name for logging
    
    Returns:
        (results_list, avg_precision_at_k, avg_recall, avg_mrr, avg_ap)
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {retriever_name}")
    print(f"{'='*70}")
    
    results = []
    precisions = []
    recalls = []
    mrrs = []
    aps = []
    
    for idx, query_item in enumerate(query_set, 1):
        question = query_item.get("question", "")
        ground_truth_refs = query_item.get("references", [])
        
        # Extract ground truth file names
        gt_files = [ref.get("file", "") for ref in ground_truth_refs]
        gt_files = [f for f in gt_files if f]  # Filter empty
        
        if not gt_files:
            print(f"[{idx}] Query: {question[:60]}...")
            print(f"     ⚠️  No ground truth references found, skipping")
            continue
        
        # Retrieve documents
        start_time = time.time()
        retrieved = retriever.search(question, k=k)
        elapsed = time.time() - start_time
        
        # Extract file names from retrieved results
        # Note: retriever returns "doc_id" which may contain the filename
        retrieved_files = [r.get("doc_id", "") for r in retrieved]
        
        # Calculate metrics
        precision, recall = calculate_precision_recall(retrieved_files, gt_files)
        mrr = calculate_mean_reciprocal_rank(retrieved_files, gt_files)
        ap = calculate_average_precision(retrieved_files, gt_files)
        
        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)
        aps.append(ap)
        
        # Prepare retrieved documents with full data
        retrieved_docs = [
            {
                "doc_id": r.get("doc_id", ""),
                "score": r.get("score", 0),
                "text": r.get("text", ""),
            }
            for r in retrieved
        ]
        
        result_item = {
            "query_idx": idx,
            "question": question,
            "ground_truth_files": gt_files,
            "retrieved_files": retrieved_files,
            "retrieved_documents": retrieved_docs,
            "precision_at_k": precision,
            "recall": recall,
            "mrr": mrr,
            "average_precision": ap,
            "retrieval_time_sec": elapsed,
        }
        results.append(result_item)
        
        # Logging
        status = "✅" if (precision > 0 and recall > 0) else "❌"
        print(f"[{idx}] {status} P@k: {precision:.2f} | R: {recall:.2f} | MRR: {mrr:.2f} | AP: {ap:.2f} | Time: {elapsed:.3f}s")
        print(f"     GT ({len(gt_files)} files): {gt_files}")
        print(f"     Retrieved ({len(retrieved_files)} files): {retrieved_files}")
        print()
    
    # Calculate averages
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    avg_ap = sum(aps) / len(aps) if aps else 0.0
    
    # Validate metrics are in [0, 1]
    assert 0.0 <= avg_precision <= 1.0, f"Precision out of range: {avg_precision}"
    assert 0.0 <= avg_recall <= 1.0, f"Recall out of range: {avg_recall}"
    assert 0.0 <= avg_mrr <= 1.0, f"MRR out of range: {avg_mrr}"
    assert 0.0 <= avg_ap <= 1.0, f"MAP out of range: {avg_ap}"
    
    print(f"\n{'-'*70}")
    print(f"Summary for {retriever_name}:")
    print(f"  Total Queries Evaluated: {len(results)}")
    print(f"  Average Precision@k: {avg_precision:.4f}")
    print(f"  Average Recall: {avg_recall:.4f}")
    print(f"  Mean Reciprocal Rank: {avg_mrr:.4f}")
    print(f"  Mean Average Precision: {avg_ap:.4f}")
    print(f"{'-'*70}\n")
    
    return results, avg_precision, avg_recall, avg_mrr, avg_ap


# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================

def main():
    """Main evaluation pipeline - evaluate selected retrievers."""
    
    # ======== COMMAND LINE ARGUMENTS ========
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval methods on query set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluation.py --model document_level
  python evaluation.py --model multi_layer
  python evaluation.py --model all
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["document_level", "multi_layer", "all"],
        help="Which retriever(s) to evaluate (default: all)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top results to retrieve (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file path for results (default: evaluation_results.json)"
    )
    args = parser.parse_args()
    # ========================================
    
    # Load query set
    query_set_path = "query_set.json"
    if not os.path.exists(query_set_path):
        print(f"❌ Error: {query_set_path} not found")
        sys.exit(1)
    
    with open(query_set_path, "r", encoding="utf-8") as f:
        query_set = json.load(f)
    
    print(f"📚 Loaded {len(query_set)} queries from {query_set_path}\n")
    
    # ======== DEFINE ALL AVAILABLE RETRIEVERS ========
    all_retrievers = {
        "document_level": {
            "class": EmbeddingDocumentRetriever,
            "name": "Document-Level Embedding (BGE-M3)",
            "key": "document_level",
        },
        "multi_layer": {
            "class": MultiLayerEmbeddingRetriever,
            "name": "Multi-Layer Embedding (BGE-M3)",
            "key": "multi_layer",
        },
    }
    # ==================================================
    
    # Select retrievers based on args
    if args.model == "all":
        retrievers_to_eval = list(all_retrievers.values())
    else:
        retrievers_to_eval = [all_retrievers[args.model]]
    
    print(f"Evaluating: {', '.join([r['name'] for r in retrievers_to_eval])}")
    print(f"Top-k: {args.k}")
    print(f"Output file: {args.output}\n")
    
    k = args.k
    evaluation_output = {
        "metadata": {
            "query_set_file": query_set_path,
            "total_queries": len(query_set),
            "k_retrieved": k,
            "models_evaluated": [r['name'] for r in retrievers_to_eval],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    for retriever_config in retrievers_to_eval:
        print(f"\n{'='*70}")
        print(f"Initializing: {retriever_config['name']}")
        print(f"{'='*70}")
        
        # Initialize retriever with correct paths
        # Get the directory where evaluation.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_folder = os.path.join(base_dir, "ntu_rules_pdfs")
        
        retriever_instance = retriever_config["class"](
            pdf_folder=pdf_folder,
            corpus_path=os.path.join(base_dir, "embedding", f"{retriever_config['key']}_corpus.json"),
            index_path=os.path.join(base_dir, "embedding", f"{retriever_config['key']}.index"),
        )
        retriever_instance.build_or_load_corpus_and_index()
        
        # Evaluate
        results, avg_precision, avg_recall, avg_mrr, avg_ap = evaluate_retriever(
            retriever_instance,
            query_set,
            k=k,
            retriever_name=retriever_config['name']
        )
        
        # Store results
        evaluation_output[retriever_config["key"]] = {
            "name": retriever_config['name'],
            "average_precision_at_k": avg_precision,
            "average_recall": avg_recall,
            "mean_reciprocal_rank": avg_mrr,
            "mean_average_precision": avg_ap,
            "results": results,
        }
    
    # Save results
    output_file = args.output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {output_file}\n")
    
    # Print summary
    if len(retrievers_to_eval) > 1:
        print("="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Metric':<30}", end="")
        for retriever in retrievers_to_eval:
            print(f" {retriever['name']:<25}", end="")
        print()
        print("-"*70)
        
        print(f"{'Average Precision':<30}", end="")
        for retriever in retrievers_to_eval:
            key = retriever['key']
            precision = evaluation_output[key]['average_precision']
            print(f" {precision:<25.4f}", end="")
        print()
        
        print(f"{'Average Recall':<30}", end="")
        for retriever in retrievers_to_eval:
            key = retriever['key']
            recall = evaluation_output[key]['average_recall']
            print(f" {recall:<25.4f}", end="")
        print()
        
        print(f"{'Average F1-Score':<30}", end="")
        for retriever in retrievers_to_eval:
            key = retriever['key']
            f1 = evaluation_output[key]['average_f1']
            print(f" {f1:<25.4f}", end="")
        print()
        print("="*70)
    else:
        print("="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        key = retrievers_to_eval[0]['key']
        summary = evaluation_output[key]
        print(f"Method: {summary['name']}")
        print(f"Average Precision@k: {summary['average_precision_at_k']:.4f}")
        print(f"Average Recall: {summary['average_recall']:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {summary['mean_reciprocal_rank']:.4f}")
        print(f"Mean Average Precision (MAP): {summary['mean_average_precision']:.4f}")
        print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
