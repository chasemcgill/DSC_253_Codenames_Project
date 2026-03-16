"""
embeddings.py — GloVe loading, similarity utilities, and FastEmbeddings matrix class.

Dict-based functions (load_glove, cosine_similarity, etc.) are kept for
compatibility. FastEmbeddings is the recommended interface for cluers because
it uses a precomputed matrix and BLAS matrix-vector multiplies that are
20-50x faster than the dict-based equivalents.

Also contains is_valid_clue() for filtering candidate clue words.
"""

from __future__ import annotations

import numpy as np
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# Vocabulary / clue-validity helpers
# ---------------------------------------------------------------------------

def get_stem(word: str) -> str:
    return _stemmer.stem(word.lower())


def is_valid_clue(candidate: str, board_words: list[str]) -> bool:
    """
    Return True if candidate is a legal clue — not on the board and not a
    morphological variant (same stem) of any board word.
    """
    candidate_lower = candidate.lower()
    candidate_stem = get_stem(candidate)
    for board_word in board_words:
        if candidate_lower == board_word.lower():
            return False
        if candidate_stem == get_stem(board_word):
            return False
    return True


# ---------------------------------------------------------------------------
# Dict-based embedding helpers (slow but simple)
# ---------------------------------------------------------------------------

def load_glove(filepath: str) -> dict[str, np.ndarray]:
    """Load all GloVe vectors from file into a word→vector dict."""
    embeddings: dict[str, np.ndarray] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return embeddings


def load_glove_filtered(filepath: str, max_words: int = 50_000) -> dict[str, np.ndarray]:
    """
    Load only the top max_words most frequent words from a GloVe file.
    GloVe files are sorted by descending frequency, so the first max_words
    lines give the most common words — large enough for good clues while
    cutting search time ~8x vs. the full 400k-word file.
    """
    embeddings: dict[str, np.ndarray] = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            embeddings[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return embeddings


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    return float(dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def compute_centroid(words: list[str], embeddings: dict) -> np.ndarray | None:
    vectors = [embeddings[w] for w in words if w in embeddings]
    return np.mean(vectors, axis=0) if vectors else None


def most_similar_to_vector(
    target_vec: np.ndarray,
    embeddings: dict,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Find k most similar words to a given vector (dict-based, slow)."""
    sims = [(w, cosine_similarity(target_vec, v)) for w, v in embeddings.items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def most_similar(word: str, embeddings: dict, k: int = 10) -> list[tuple[str, float]]:
    if word not in embeddings:
        return []
    return most_similar_to_vector(embeddings[word], embeddings, k=k)


# ---------------------------------------------------------------------------
# FastEmbeddings — vectorised matrix class (recommended)
# ---------------------------------------------------------------------------

class FastEmbeddings:
    """
    Precomputed GloVe matrix for fast nearest-neighbour search.

    Stores all vectors as a single numpy matrix and precomputes L2 norms so
    that most_similar_to_vector can use a single BLAS matrix-vector multiply
    instead of a Python loop — 20-50x faster than the dict-based version.

    Implements __contains__ and __getitem__ so it can be passed directly to
    any function that expects a {word: vector} dict.

    Attributes
    ----------
    words       : list[str]         — words in file order (index → word)
    word_to_idx : dict[str, int]    — word → row index in matrix
    matrix      : np.ndarray (N, D) — all word vectors stacked row-wise
    norms       : np.ndarray (N,)   — precomputed L2 norm of each row
    """

    def __init__(self, filepath: str, max_words: int = 50_000) -> None:
        words: list[str] = []
        vectors: list[np.ndarray] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                words.append(parts[0])
                vectors.append(np.array(parts[1:], dtype=np.float32))

        self.words: list[str] = words
        self.word_to_idx: dict[str, int] = {w: i for i, w in enumerate(words)}
        self.matrix: np.ndarray = np.stack(vectors)
        raw_norms = np.linalg.norm(self.matrix, axis=1)
        self.norms: np.ndarray = np.where(raw_norms == 0, 1.0, raw_norms)

    # --- dict-compatible interface ---

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx

    def __getitem__(self, word: str) -> np.ndarray:
        return self.matrix[self.word_to_idx[word]]

    def __len__(self) -> int:
        return len(self.words)

    def keys(self):
        return self.words

    def items(self):
        for i, word in enumerate(self.words):
            yield word, self.matrix[i]

    # --- core operations ---

    def get_vector(self, word: str) -> np.ndarray | None:
        idx = self.word_to_idx.get(word)
        return self.matrix[idx] if idx is not None else None

    def cosine_similarity(self, word1: str, word2: str) -> float:
        idx1 = self.word_to_idx.get(word1)
        idx2 = self.word_to_idx.get(word2)
        if idx1 is None or idx2 is None:
            return 0.0
        return float(
            np.dot(self.matrix[idx1], self.matrix[idx2])
            / (self.norms[idx1] * self.norms[idx2])
        )

    def compute_centroid(self, words: list[str]) -> np.ndarray | None:
        indices = [self.word_to_idx[w] for w in words if w in self.word_to_idx]
        return np.mean(self.matrix[indices], axis=0) if indices else None

    def most_similar_to_vector(
        self,
        target_vec: np.ndarray,
        k: int = 50,
        exclude_words: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k vocabulary words most similar to target_vec.

        Uses a single matrix-vector multiply (BLAS SGEMV) and np.argpartition
        for O(N) top-k selection — 20-50x faster than the dict-based version.
        """
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return []
        target_normalized = target_vec / target_norm

        dot_products = self.matrix @ target_normalized
        similarities = dot_products / self.norms

        effective_k = min(k, len(self.words))
        top_k_idx = np.argpartition(similarities, -effective_k)[-effective_k:]
        top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]

        results: list[tuple[str, float]] = []
        for idx in top_k_idx:
            word = self.words[idx]
            if exclude_words and word in exclude_words:
                continue
            results.append((word, float(similarities[idx])))
        return results

    def score_clue_fast(
        self,
        clue_word: str,
        target_words: list[str],
        opponent_words: list[str],
        assassin: str,
        assassin_penalty: float = 2.0,
    ) -> float:
        """
        Score a candidate clue against the board using numpy fancy indexing.

        Scoring logic:  min_target_sim - max_opponent_sim - penalty * assassin_sim
        Returns float('-inf') if clue or all target words are missing.
        """
        clue_idx = self.word_to_idx.get(clue_word)
        if clue_idx is None:
            return float("-inf")

        clue_vec = self.matrix[clue_idx]
        clue_norm = self.norms[clue_idx]

        t_idx = [self.word_to_idx[w] for w in target_words if w in self.word_to_idx]
        if not t_idx:
            return float("-inf")
        min_target_sim = float(
            np.min((self.matrix[t_idx] @ clue_vec) / (self.norms[t_idx] * clue_norm))
        )

        o_idx = [self.word_to_idx[w] for w in opponent_words if w in self.word_to_idx]
        max_opponent_sim = (
            float(np.max((self.matrix[o_idx] @ clue_vec) / (self.norms[o_idx] * clue_norm)))
            if o_idx else 0.0
        )

        a_idx = self.word_to_idx.get(assassin)
        assassin_sim = (
            float(np.dot(self.matrix[a_idx], clue_vec) / (self.norms[a_idx] * clue_norm))
            if a_idx is not None else 0.0
        )

        return min_target_sim - max_opponent_sim - (assassin_penalty * assassin_sim)
