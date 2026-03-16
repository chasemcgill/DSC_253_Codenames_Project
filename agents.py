"""
agents.py — All Codenames cluers and guessers.

Includes:
  - BaseCluer / BaseGuesser  — abstract interfaces for teammates to subclass
  - CentroidCluer            — fast centroid-based cluer (recommended)
  - CentroidCluerSlow        — dict-based reference implementation
  - SimpleCluer              — single-word baseline cluer
  - RandomCluer              — random noise-floor cluer
  - SimilarityGuesser        — cosine similarity guesser
  - SimpleGuesser            — always guesses one word (conservative baseline)
  - RandomGuesser            — random noise-floor guesser

Algorithm (CentroidCluer):
  For every 2-4 word subset of our remaining words, compute the centroid
  vector and find the vocabulary word that maximises:
      min_target_similarity - max_opponent_similarity - assassin_penalty
  Uses FastEmbeddings matrix operations for a 20-50x speedup over the
  dict-based version.

Usage:
  from embeddings import FastEmbeddings
  from agents import CentroidCluer, SimilarityGuesser

  emb = FastEmbeddings("glove.6B/glove.6B.300d.txt")
  cluer  = CentroidCluer(emb)
  guesser = SimilarityGuesser(emb)
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from itertools import combinations

from board import BoardState
from embeddings import (
    FastEmbeddings,
    cosine_similarity,
    compute_centroid,
    most_similar_to_vector,
    is_valid_clue,
)


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class BaseCluer(ABC):
    """
    Interface every cluer must implement.

    give_clue receives the full board state and returns (clue_word, number) where:
      - clue_word is a single English word NOT currently on the board
      - number    is how many board words the clue is intended to target
    """

    @abstractmethod
    def give_clue(self, board_state: BoardState) -> tuple[str | None, int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class BaseGuesser(ABC):
    """
    Interface every guesser must implement.

    make_guesses receives the clue word, the declared number, and the list of
    words still on the board. Returns an ordered list of at most ``number``
    guesses (most confident first).
    """

    @abstractmethod
    def make_guesses(
        self, clue: str, number: int, remaining_words: list[str]
    ) -> list[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Core algorithm helpers
# ---------------------------------------------------------------------------

def _generate_subsets(
    words: list[str], min_size: int = 2, max_size: int = 4
) -> list[list[str]]:
    """All subsets of words with length in [min_size, max_size]."""
    return [
        list(subset)
        for size in range(min_size, max_size + 1)
        for subset in combinations(words, size)
    ]


def _score_clue(
    clue_vec,
    target_words: list[str],
    opponent_words: list[str],
    assassin: str,
    embeddings: dict,
    assassin_penalty: float = 2.0,
) -> float:
    """
    Score a candidate clue vector (dict-based, used by CentroidCluerSlow).
    Higher is better: min_target_sim - max_opponent_sim - penalty * assassin_sim.
    """
    target_sims = [
        cosine_similarity(clue_vec, embeddings[w])
        for w in target_words if w in embeddings
    ]
    if not target_sims:
        return float("-inf")

    opponent_sims = [
        cosine_similarity(clue_vec, embeddings[w])
        for w in opponent_words if w in embeddings
    ]
    max_opponent_sim = max(opponent_sims) if opponent_sims else 0.0

    assassin_sim = (
        cosine_similarity(clue_vec, embeddings[assassin])
        if assassin in embeddings else 0.0
    )

    return min(target_sims) - max_opponent_sim - (assassin_penalty * assassin_sim)


def _find_best_clue(
    our_words: list[str],
    opponent_words: list[str],
    assassin: str,
    embeddings: dict,
    num_candidates: int = 50,
) -> tuple[str | None, int, list[str], float]:
    """
    Dict-based centroid cluer (slow reference implementation).
    Returns (clue_word, number, target_words, score).
    """
    all_board_words = our_words + opponent_words + [assassin]
    subsets = _generate_subsets(our_words)

    best_clue, best_score, best_targets = None, float("-inf"), None

    for subset in subsets:
        centroid = compute_centroid(subset, embeddings)
        if centroid is None:
            continue
        for candidate_word, _ in most_similar_to_vector(centroid, embeddings, k=num_candidates):
            if not is_valid_clue(candidate_word, all_board_words):
                continue
            score = _score_clue(
                embeddings[candidate_word], subset, opponent_words, assassin, embeddings
            )
            if score > best_score:
                best_score, best_clue, best_targets = score, candidate_word, subset

    if best_clue is None:
        return None, 0, [], 0
    return best_clue, len(best_targets), best_targets, best_score


def _find_best_clue_fast(
    our_words: list[str],
    opponent_words: list[str],
    assassin: str,
    fast_emb: FastEmbeddings,
    num_candidates: int = 50,
) -> tuple[str | None, int, list[str], float]:
    """
    Vectorised centroid cluer using FastEmbeddings matrix operations.

    Same logic as _find_best_clue but replaces every inner loop with a
    single BLAS matrix-vector multiply — 20-50x faster.
    Returns (clue_word, number, target_words, score).
    """
    all_board_words = our_words + opponent_words + [assassin]
    board_word_set = {w.lower() for w in all_board_words}
    subsets = _generate_subsets(our_words)

    best_clue, best_score, best_targets = None, float("-inf"), None

    for subset in subsets:
        centroid = fast_emb.compute_centroid(subset)
        if centroid is None:
            continue
        candidates = fast_emb.most_similar_to_vector(
            centroid, k=num_candidates, exclude_words=board_word_set
        )
        for candidate_word, _ in candidates:
            if not is_valid_clue(candidate_word, all_board_words):
                continue
            score = fast_emb.score_clue_fast(
                candidate_word, subset, opponent_words, assassin
            )
            if score > best_score:
                best_score, best_clue, best_targets = score, candidate_word, subset

    if best_clue is None:
        return None, 0, [], 0
    return best_clue, len(best_targets), best_targets, best_score


# ---------------------------------------------------------------------------
# Cluer implementations
# ---------------------------------------------------------------------------

class CentroidCluer(BaseCluer):
    """
    Fast centroid cluer using FastEmbeddings matrix operations (recommended).

    Searches all 2-4 word subsets of our remaining words, computes their
    centroid vector, and finds the vocabulary word that maximises:
        min_target_similarity - max_opponent_similarity - assassin_penalty

    Parameters
    ----------
    embeddings     : a loaded FastEmbeddings instance
    num_candidates : top-k neighbours to score per subset (default 50)
    """

    def __init__(self, embeddings: FastEmbeddings, num_candidates: int = 50) -> None:
        self.embeddings = embeddings
        self.num_candidates = num_candidates

    def give_clue(self, board_state: BoardState) -> tuple[str | None, int]:
        clue, number, _targets, _score = _find_best_clue_fast(
            board_state.remaining_our,
            board_state.remaining_opponent,
            board_state.assassin,
            self.embeddings,
            num_candidates=self.num_candidates,
        )
        return clue, number


class CentroidCluerSlow(BaseCluer):
    """
    Reference cluer using a plain dict and Python loops.
    Functionally identical to CentroidCluer but 20-50x slower.
    Kept for benchmarking and comparison.

    Parameters
    ----------
    embeddings     : dict loaded via load_glove_filtered
    num_candidates : top-k neighbours to score per subset (default 50)
    """

    def __init__(self, embeddings: dict, num_candidates: int = 50) -> None:
        self.embeddings = embeddings
        self.num_candidates = num_candidates

    def give_clue(self, board_state: BoardState) -> tuple[str | None, int]:
        clue, number, _targets, _score = _find_best_clue(
            board_state.remaining_our,
            board_state.remaining_opponent,
            board_state.assassin,
            self.embeddings,
            num_candidates=self.num_candidates,
        )
        return clue, number


class SimpleCluer(BaseCluer):
    """
    Baseline cluer: for each of our words, finds its single most similar
    valid clue. Returns the best such clue with number=1.
    No multi-word coverage and no danger penalty — deliberately weak baseline.
    """

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def give_clue(self, board_state: BoardState) -> tuple[str | None, int]:
        our, opp, assassin = (
            board_state.remaining_our,
            board_state.remaining_opponent,
            board_state.assassin,
        )
        all_board_words = our + opp + [assassin]
        best_clue, best_score, best_target = None, float("-inf"), None

        for target in our:
            if target not in self.embeddings:
                continue
            if hasattr(self.embeddings, "most_similar_to_vector"):
                candidates = self.embeddings.most_similar_to_vector(
                    self.embeddings[target], k=30
                )
            else:
                candidates = most_similar_to_vector(
                    self.embeddings[target], self.embeddings, k=30
                )
            for word, sim in candidates:
                if not is_valid_clue(word, all_board_words):
                    continue
                if sim > best_score:
                    best_score, best_clue, best_target = sim, word, target
                break  # only top valid candidate per target

        if best_clue is None:
            return None, 0
        return best_clue, 1


class RandomCluer(BaseCluer):
    """
    Baseline cluer: picks a random valid word and a random number.
    Useful as a noise floor for evaluation.
    """

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def give_clue(self, board_state: BoardState) -> tuple[str | None, int]:
        our, opp, assassin = (
            board_state.remaining_our,
            board_state.remaining_opponent,
            board_state.assassin,
        )
        all_board_words = our + opp + [assassin]
        vocab_sample = random.sample(
            list(self.embeddings.keys()), min(5_000, len(self.embeddings))
        )
        valid = [w for w in vocab_sample if is_valid_clue(w, all_board_words)]
        if not valid:
            return None, 0
        return random.choice(valid), random.randint(1, min(3, len(our)))


# ---------------------------------------------------------------------------
# Guesser implementations
# ---------------------------------------------------------------------------

class SimilarityGuesser(BaseGuesser):
    """
    Ranks all remaining words by cosine similarity to the clue vector and
    returns the top ``number`` matches.
    """

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def make_guesses(
        self, clue: str, number: int, remaining_words: list[str]
    ) -> list[str]:
        if clue not in self.embeddings:
            return remaining_words[:number]
        clue_vec = self.embeddings[clue]
        sims = [
            (w, cosine_similarity(clue_vec, self.embeddings[w]) if w in self.embeddings else 0.0)
            for w in remaining_words
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in sims[:number]]


class SimpleGuesser(BaseGuesser):
    """
    Always returns exactly one word — the most similar to the clue —
    regardless of the given number. Demonstrates the cost of overly
    conservative guessing.
    """

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def make_guesses(
        self, clue: str, number: int, remaining_words: list[str]
    ) -> list[str]:
        if not remaining_words:
            return []
        if clue not in self.embeddings:
            return [random.choice(remaining_words)]
        clue_vec = self.embeddings[clue]
        best = max(
            remaining_words,
            key=lambda w: cosine_similarity(clue_vec, self.embeddings[w])
            if w in self.embeddings else -1.0,
        )
        return [best]


class RandomGuesser(BaseGuesser):
    """Picks ``number`` words uniformly at random from remaining board words."""

    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings

    def make_guesses(
        self, clue: str, number: int, remaining_words: list[str]
    ) -> list[str]:
        if not remaining_words:
            return []
        return random.sample(remaining_words, min(number, len(remaining_words)))
