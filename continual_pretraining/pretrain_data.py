"""PDF preprocessing for continual pretraining.

- text extraction using PyMuPDF for layout-preserving text
- optional pdfplumber-based extraction for better tables/columns
- simple header/footer detection and removal using repeated text heuristics
- cleaning and normalization utilities
- chunking into training-ready documents and saving as JSONL with metadata

Usage (example):
	from continual_pretraining import pretrain_data as pd
	pd.process_folder("./ntu_rules_pdfs", "./pretrain_corpus.jsonl")

This module is intentionally dependency-tolerant: pdfplumber is optional.
"""

from __future__ import annotations

import os
import re
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

try:
	import fitz  # PyMuPDF
except Exception as e:
	raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf") from e

try:
	import pdfplumber
	_HAS_PDFPLUMBER = True
except Exception:
	pdfplumber = None
	_HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PageText:
	page_number: int
	text: str


def extract_text_fitz(path: str) -> List[PageText]:
	"""Extract text per page using PyMuPDF (fitz).

	Returns a list of PageText preserving the page ordering.
	"""
	doc = fitz.open(path)
	pages = []
	for i in range(doc.page_count):
		page = doc.load_page(i)
		# get_text("blocks") preserves blocks and can be joined to retain some layout
		text = page.get_text("text")
		pages.append(PageText(page_number=i + 1, text=text))
	doc.close()
	return pages


def extract_text_pdfplumber(path: str) -> List[PageText]:
	"""Extract text per page using pdfplumber (better for tables/columns).

	Falls back to PyMuPDF if pdfplumber is not installed.
	"""
	if not _HAS_PDFPLUMBER:
		logger.warning("pdfplumber not installed; falling back to PyMuPDF extractor")
		return extract_text_fitz(path)

	pages = []
	with pdfplumber.open(path) as pdf:
		for i, page in enumerate(pdf.pages):
			# pdfplumber's extract_text preserves columns better in many cases
			txt = page.extract_text() or ""
			pages.append(PageText(page_number=i + 1, text=txt))
	return pages


def detect_repeat_headers_footers(pages: List[PageText], top_n: int = 3, min_repeat: int = 2) -> Tuple[set, set]:
	"""Heuristic: gather candidate header/footer lines that repeat across pages.

	We look at the first and last N lines of each page and count repetitions.
	Returns two sets: headers, footers (strings to remove).
	"""
	top_lines = []
	bottom_lines = []
	for p in pages:
		lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
		if not lines:
			continue
		top_lines.extend(lines[:top_n])
		bottom_lines.extend(lines[-top_n:])

	top_counts = Counter(top_lines)
	bottom_counts = Counter(bottom_lines)

	headers = {s for s, c in top_counts.items() if c >= min_repeat and len(s) > 3}
	footers = {s for s, c in bottom_counts.items() if c >= min_repeat and len(s) > 3}
	logger.debug("Detected headers: %s", headers)
	logger.debug("Detected footers: %s", footers)
	return headers, footers


def remove_headers_footers_from_page(text: str, headers: set, footers: set) -> str:
	"""Remove header/footer strings from page text conservatively.

	Matches whole lines exactly (after strip), and removes trailing page numbers like 'Page 3' heuristically.
	"""
	lines = [ln.rstrip() for ln in text.splitlines()]
	out_lines = []
	for ln in lines:
		s = ln.strip()
		if s in headers or s in footers:
			continue
		# remove simple page number patterns
		if re.fullmatch(r"(page\s*)?\d+", s, flags=re.IGNORECASE):
			continue
		out_lines.append(ln)
	return "\n".join(out_lines)


def clean_text_block(text: str) -> str:
	"""Apply generic cleaning rules: normalize whitespace, remove multiple newlines, fix hyphenation."""
	# Fix common hyphenation where words are split at line end: 'exam-\nple' -> 'example'
	text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
	# Normalize newlines and remove excessive blank lines
	text = re.sub(r"\n{3,}", "\n\n", text)
	# Strip trailing/leading whitespace per line
	lines = [ln.strip() for ln in text.splitlines()]
	# Remove lines that are just non-informational artifacts (e.g., sequences of symbols)
	filtered = [ln for ln in lines if not re.fullmatch(r"[-=*_]{2,}", ln)]
	# Join with single newline
	cleaned = "\n".join(filtered)
	# Collapse multiple spaces
	cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
	cleaned = cleaned.strip()
	return cleaned


def pages_to_document(pages: List[PageText], detect_headers: bool = True) -> str:
	"""Convert list of PageText to a single cleaned document string.

	If detect_headers is True we attempt to detect and strip repeating headers/footers.
	"""
	headers, footers = (set(), set())
	if detect_headers:
		headers, footers = detect_repeat_headers_footers(pages)

	cleaned_pages = []
	for p in pages:
		txt = remove_headers_footers_from_page(p.text, headers, footers)
		txt = clean_text_block(txt)
		if txt:
			cleaned_pages.append(f"\n\n[PAGE {p.page_number}]\n" + txt)

	return "\n\n".join(cleaned_pages)


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
	"""Simple chunker based on word counts (not tokens). For pretraining use, tune params.

	This is a rough approximation: divides text into chunks of ~max_tokens words with overlap.
	"""
	words = text.split()
	if not words:
		return []
	chunks = []
	start = 0
	while start < len(words):
		end = min(start + max_tokens, len(words))
		chunk = " ".join(words[start:end])
		chunks.append(chunk)
		if end == len(words):
			break
		start = end - overlap
	return chunks


def save_jsonl(documents: List[Dict], out_path: str):
	"""Save list of dicts as JSONL to out_path.

	Each dict should be JSON-serializable. Overwrites existing file.
	"""
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		for doc in documents:
			f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def process_pdf(path: str, use_pdfplumber: bool = False, chunk_words: int = 512, overlap: int = 50) -> List[Dict]:
	"""Process a single PDF and return a list of JSON-serializable document chunks.

	Each chunk dict contains: id, source, page_range, text, and optional metadata.
	"""
	extractor = extract_text_pdfplumber if (use_pdfplumber and _HAS_PDFPLUMBER) else extract_text_fitz
	pages = extractor(path)
	if not pages:
		return []

	full_text = pages_to_document(pages, detect_headers=True)

	# Word-based chunking (default preprocessing path)
	chunks = chunk_text(full_text, max_tokens=chunk_words, overlap=overlap)

	docs = []
	base = os.path.basename(path)
	for i, c in enumerate(chunks, start=1):
		doc = {
			"id": f"{base}::chunk_{i}",
			"source": path,
			"chunk_index": i,
			"text": c,
		}
		docs.append(doc)
	return docs


def process_folder(folder: str, out_jsonl: str, use_pdfplumber: bool = False, recursive: bool = False,
				   chunk_words: int = 512, overlap: int = 50) -> int:
	"""Process all PDFs in folder and save results to out_jsonl. Returns number of chunks written.

	If recursive is True, walk subdirectories.
	"""
	files = []
	if recursive:
		for root, _, fnames in os.walk(folder):
			for f in fnames:
				if f.lower().endswith(".pdf"):
					files.append(os.path.join(root, f))
	else:
		for f in os.listdir(folder):
			if f.lower().endswith(".pdf"):
				files.append(os.path.join(folder, f))

	all_docs = []
	for p in sorted(files):
		logger.info("Processing %s", p)
		try:
			docs = process_pdf(p, use_pdfplumber=use_pdfplumber, chunk_words=chunk_words, overlap=overlap)
			all_docs.extend(docs)
		except Exception as e:
			logger.exception("Failed to process %s: %s", p, e)

	save_jsonl(all_docs, out_jsonl)
	logger.info("Wrote %d document chunks to %s", len(all_docs), out_jsonl)
	return len(all_docs)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Extract and preprocess PDFs into JSONL for pretraining")
	parser.add_argument("folder", help="Folder containing PDFs to process")
	parser.add_argument("out", help="Output JSONL path")
	parser.add_argument("--pdfplumber", action="store_true", help="Use pdfplumber extractor when available")
	parser.add_argument("--recursive", action="store_true", help="Search folders recursively")
	parser.add_argument("--chunk_words", type=int, default=512, help="Approx words per chunk")
	parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks")
	
	args = parser.parse_args()
	process_folder(args.folder, args.out, use_pdfplumber=args.pdfplumber, recursive=args.recursive,
				   chunk_words=args.chunk_words, overlap=args.overlap)

