"""
Retrieval engine for MineGPT.

Loads the DGMS knowledge base from the `data/` folder, splits each document into
overlapping text chunks, builds a TF-IDF index over those chunks, and retrieves
the most relevant chunks for a user query using cosine similarity.

This is intentionally lightweight (scikit-learn only) so it installs quickly and
runs fully offline.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_PATH = os.path.join(PROJECT_ROOT, "minegpt_index.pkl")


@dataclass
class Chunk:
    """A single retrievable passage of the knowledge base."""

    text: str
    source: str  # file name the chunk came from
    heading: str  # nearest markdown heading (for citation / context)


@dataclass
class Retriever:
    """TF-IDF based retriever over the DGMS knowledge base."""

    chunks: list[Chunk] = field(default_factory=list)
    vectorizer: TfidfVectorizer | None = None
    matrix: object = None  # scipy sparse matrix of TF-IDF vectors

    # -- Building -----------------------------------------------------------
    @staticmethod
    def _split_into_chunks(text: str, source: str,
                           max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
        """Split a document into overlapping chunks, tracking the nearest heading."""
        lines = text.splitlines()
        chunks: list[Chunk] = []
        current_heading = source
        buffer: list[str] = []

        def flush(buf: list[str], heading: str) -> None:
            body = "\n".join(buf).strip()
            if not body:
                return
            # Further split long paragraphs with a sliding window.
            if len(body) <= max_chars:
                chunks.append(Chunk(text=body, source=source, heading=heading))
                return
            start = 0
            while start < len(body):
                piece = body[start:start + max_chars]
                chunks.append(Chunk(text=piece.strip(), source=source, heading=heading))
                start += max_chars - overlap

        for line in lines:
            heading_match = re.match(r"^#{1,6}\s+(.*)", line)
            if heading_match:
                # New heading -> close the previous section.
                flush(buffer, current_heading)
                buffer = []
                current_heading = heading_match.group(1).strip()
            else:
                buffer.append(line)
        flush(buffer, current_heading)
        return chunks

    def build(self, data_dir: str = DATA_DIR) -> "Retriever":
        """Read every .md/.txt file under data_dir and build the TF-IDF index."""
        paths = sorted(
            glob.glob(os.path.join(data_dir, "*.md"))
            + glob.glob(os.path.join(data_dir, "*.txt"))
        )
        if not paths:
            raise FileNotFoundError(
                f"No .md or .txt documents found in {data_dir}. "
                "Add DGMS documents to the data/ folder first."
            )

        self.chunks = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            source = os.path.basename(path)
            self.chunks.extend(self._split_into_chunks(text, source))

        corpus = [f"{c.heading}\n{c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        return self

    # -- Persistence --------------------------------------------------------
    def save(self, path: str = INDEX_PATH) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix},
                fh,
            )

    @classmethod
    def load(cls, path: str = INDEX_PATH) -> "Retriever":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        return cls(chunks=data["chunks"], vectorizer=data["vectorizer"], matrix=data["matrix"])

    # -- Querying -----------------------------------------------------------
    def search(self, query: str, top_k: int = 4, min_score: float = 0.03
               ) -> list[tuple[Chunk, float]]:
        """Return up to top_k (chunk, score) pairs most relevant to the query."""
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q_vec = self.vectorizer.transform([query])
        # TfidfVectorizer L2-normalizes rows by default, so cosine similarity is
        # just the dot product. Computing it directly avoids sklearn.metrics
        # (whose compiled DLL is blocked by some Windows app-control policies).
        scores = np.asarray((q_vec @ self.matrix.T).todense()).ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        results: list[tuple[Chunk, float]] = []
        for idx in ranked:
            score = float(scores[idx])
            if score >= min_score:
                results.append((self.chunks[idx], score))
        return results


def ensure_index(rebuild: bool = False) -> Retriever:
    """Load the saved index, or build and save it if missing / rebuild requested."""
    if not rebuild and os.path.exists(INDEX_PATH):
        try:
            return Retriever.load()
        except Exception:
            pass  # fall through to rebuild on any load error
    retriever = Retriever().build()
    retriever.save()
    return retriever
