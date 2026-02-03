import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:  # optional dependency
    OpenAI = None


@dataclass
class RetrievedChunk:
    text: str
    score: float
    metadata: dict


def load_pdfs(paths: Iterable[str]) -> List[Tuple[str, dict]]:
    documents = []
    for path in paths:
        reader = PdfReader(path)
        text = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text.append(page_text)
            documents.append(
                (
                    page_text,
                    {
                        "source": os.path.basename(path),
                        "page": page_number,
                    },
                )
            )
    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(words):
            break
    return chunks


class RAGChatbot:
    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 120,
        top_k: int = 4,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.model = model
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_texts: List[str] = []
        self.chunk_metadata: List[dict] = []
        self.chunk_matrix: Optional[np.ndarray] = None

    def index(self, pdf_paths: Sequence[str]) -> None:
        raw_docs = load_pdfs(pdf_paths)
        chunks = []
        metadata = []
        for text, meta in raw_docs:
            for chunk in chunk_text(text, self.chunk_size, self.overlap):
                if chunk.strip():
                    chunks.append(chunk)
                    metadata.append(meta)
        if not chunks:
            raise ValueError("No text extracted from provided PDFs.")
        self.chunk_texts = chunks
        self.chunk_metadata = metadata
        self.chunk_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        if self.chunk_matrix is None:
            raise RuntimeError("Index is empty. Call index() first.")
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.chunk_matrix).flatten()
        best_indices = np.argsort(scores)[::-1][: self.top_k]
        results = []
        for idx in best_indices:
            results.append(
                RetrievedChunk(
                    text=self.chunk_texts[idx],
                    score=float(scores[idx]),
                    metadata=self.chunk_metadata[idx],
                )
            )
        return results

    def _build_prompt(self, query: str, retrieved: Sequence[RetrievedChunk]) -> str:
        context_blocks = []
        for chunk in retrieved:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            context_blocks.append(f"Source: {source} (page {page})\n{chunk.text}")
        context = "\n\n".join(context_blocks)
        return (
            "You are a helpful assistant. Answer the question using only the context below. "
            "If the answer is not present, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )

    def _answer_with_openai(self, prompt: str) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package not installed.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def chat(self, query: str) -> Tuple[str, List[RetrievedChunk]]:
        retrieved = self.retrieve(query)
        prompt = self._build_prompt(query, retrieved)
        try:
            answer = self._answer_with_openai(prompt)
        except RuntimeError:
            fallback_context = "\n\n".join(chunk.text for chunk in retrieved)
            answer = (
                "OpenAI API key not configured. Here's the most relevant context:\n\n"
                f"{fallback_context}"
            )
        return answer, retrieved
