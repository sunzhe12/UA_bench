"""
Metric engine for hierarchical classification.

Metrics:/n  - L1/L2/L3 accuracy, per-class P/R/F1, macro F1, confusion matrix
  - L2-given-L1, L3-given-L2
  - reasoning quality: domain term match rate
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# ============================================================
# Prediction data structures
# ============================================================

@dataclass
class HierPrediction:
    """Hierarchical classification prediction for one sample."""
    sample_id: str
    L1: str                      # "active" | "passive" | "unknown"
    L2: str                      # "pulse" | "communication" | "ship_noise" | "unknown"
    L3: str                      # "CW" | "LFM" | ... | "unknown"
    parse_tier: int              # 1 = direct hit, 2 = keyword fallback, 3 = failed
    raw_output: str = ""         # raw model output text
    parse_status: str = "not_parsed"


# ============================================================
# Metric data structures
# ============================================================

@dataclass
class LevelMetrics:
    """Metrics for a single classification level."""
    level: str                   # "L1" | "L2" | "L3"
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_labels: List[str] = field(default_factory=list)


@dataclass
class HierarchicalMetrics:
    """Complete hierarchical classification evaluation result."""
    total_samples: int = 0
    l1: Optional[LevelMetrics] = None
    l2: Optional[LevelMetrics] = None
    l3: Optional[LevelMetrics] = None
    l2_given_l1: float = 0.0
    l3_given_l2: float = 0.0
    parse_tier_dist: Dict[int, int] = field(default_factory=dict)


@dataclass
class ReasoningMetrics:
    """Reasoning quality metrics."""
    num_samples: int
    cascade_skipped: int = 0               # samples whose T3 was skipped because T1 was wrong
    # term-to-class alignment
    alignment_rate: float = 0.0            # share with should_appear >= 1 and should_not = 0
    contradiction_rate: float = 0.0        # should_not >= 1 (terms of other classes were used)
    # error taxonomy
    concept_confusion_rate: float = 0.0    # concept confusion: described this class with features of another
    vague_rate: float = 0.0                # vague / generic: no discriminative term
    term_stacking_rate: float = 0.0        # term stacking: positive and negative terms appear together


# ============================================================
# Main calculator
# ============================================================

class Scorer:
    """Calculator for hierarchical classification metrics."""

    def __init__(self, config: dict):
        tax = config["taxonomy"]

        # build the class key -> name mapping
        self.class_names = {}
        for level in ["L1", "L2", "L3"]:
            for c in tax[level]["classes"]:
                self.class_names[c["key"]] = c["name"]

        # class keys for L1/L2/L3 (in config order)
        self.l1_keys = [c["key"] for c in tax["L1"]["classes"]]
        self.l2_keys = [c["key"] for c in tax["L2"]["classes"]]
        self.l3_keys = [c["key"] for c in tax["L3"]["classes"]]



    def compute_level(
        self, level: str, class_keys: List[str],
        y_pred: List[str], y_true: List[str],
    ) -> LevelMetrics:
        """Compute metrics for a single classification level."""
        n = len(y_pred)
        lm = LevelMetrics(level=level, total=n, correct=0)

        # Accuracy
        lm.correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        lm.accuracy = lm.correct / n if n > 0 else 0.0

        # Per-class
        for cls in class_keys:
            tp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(y_pred, y_true) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(y_pred, y_true) if p != cls and t == cls)
            support = sum(1 for t in y_true if t == cls)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            name = self.class_names.get(cls, cls)
            lm.per_class[name] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }

        # macro F1 (only classes with support)
        f1s = [m["f1"] for m in lm.per_class.values() if m["support"] > 0]
        lm.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        # Confusion matrix (extra row/col for special labels if present)
        all_keys = list(class_keys)
        special_keys = []
        for label in ("unknown", "cascade_error"):
            if any(p == label for p in y_pred) or any(t == label for t in y_true):
                special_keys.append(label)
                all_keys.append(label)
        key_idx = {k: i for i, k in enumerate(all_keys)}
        cm = [[0] * len(all_keys) for _ in range(len(all_keys))]
        for p, t in zip(y_pred, y_true):
            pi = key_idx.get(p, -1)
            ti = key_idx.get(t, -1)
            if ti >= 0 and pi >= 0:
                cm[ti][pi] += 1
        lm.confusion_matrix = cm
        lm.confusion_labels = [self.class_names.get(k, k) for k in all_keys]

        return lm

    # ============================================================
    # Fourth part: reasoning quality
    # ============================================================

    # ================================================================
    # L3 term-to-class alignment matrix
    # imported from shared_terminology.py as the single source of truth
    # ================================================================
    from .shared_terminology import L3_SHOULD as _L3_SHOULD, get_should_not as _get_should_not
    _get_should_not = staticmethod(_get_should_not)

    _ALL_L3_TERMS = set()
    for _s in _L3_SHOULD.values():
        _ALL_L3_TERMS.update(_s)

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        """Match a phrase case-insensitively without matching inside words."""
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        return re.search(pattern, text.lower()) is not None

    @staticmethod
    def _longest_term_matches(text: str, terms) -> set:
        """Keep longest non-overlapping occurrences across the full inventory.

        Equal-length overlaps prefer the earlier occurrence, then lexical order.
        A shorter term can still match elsewhere outside an accepted span.
        """
        text_lower = text.lower()
        candidates = []
        for term in terms:
            if not term:
                continue
            pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
            for match in re.finditer(pattern, text_lower):
                candidates.append((match.start(), match.end(), term))
        candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
        accepted = []
        hits = set()
        for start, end, term in candidates:
            if any(start < other_end and other_start < end
                   for other_start, other_end in accepted):
                continue
            accepted.append((start, end))
            hits.add(term)
        return hits

    def compute_reasoning_quality(
        self,
        reasoning_texts: List[str],
        predicted_l3: List[str] = None,
    ) -> ReasoningMetrics:
        """Evaluate the quality of the reasoning text: term-to-class alignment + error taxonomy."""
        n = len(reasoning_texts)
        if n == 0:
            return ReasoningMetrics(num_samples=0)

        if predicted_l3 is None:
            predicted_l3 = [""] * n

        aligned = 0
        contradicted = 0
        confused = 0
        vague = 0
        stacked = 0

        for text, l3 in zip(reasoning_texts, predicted_l3):
            should_set = self._L3_SHOULD.get(l3, set())
            should_not_set = self._get_should_not(l3)

            hits = self._longest_term_matches(text, should_set | should_not_set)
            should_hits = hits & should_set
            should_not_hits = hits & should_not_set

            has_positive = len(should_hits) >= 1
            has_negative = len(should_not_hits) >= 1

            if has_positive and not has_negative:
                aligned += 1
            if has_negative:
                contradicted += 1

            if has_positive and has_negative:
                stacked += 1
            elif has_negative:
                confused += 1
            elif not has_positive and not has_negative:
                vague += 1

        return ReasoningMetrics(
            num_samples=n,
            alignment_rate=aligned / n,
            contradiction_rate=contradicted / n,
            concept_confusion_rate=confused / n,
            vague_rate=vague / n,
            term_stacking_rate=stacked / n,
        )

    # ============================================================
    # Bootstrap confidence intervals
    # ============================================================

    @staticmethod
    def bootstrap_ci(results: list, metric_fn, n_iter: int = 1000,
                     ci_level: float = 0.95) -> tuple:
        """Bootstrap percentile confidence interval.

        Args:
            results: list of TurnResult
            metric_fn: callable(results) → float
            n_iter: number of bootstrap iterations
            ci_level: confidence level (default 0.95)

        Returns:
            (lower, upper), or (None, None) when there are too few samples
        """
        import random as _random
        n = len(results)
        if n < 5:
            return (None, None)

        rng = _random.Random(42)
        values = []
        for _ in range(n_iter):
            sample = [results[rng.randint(0, n - 1)] for _ in range(n)]
            try:
                values.append(metric_fn(sample))
            except (ZeroDivisionError, ValueError):
                continue

        if len(values) < n_iter * 0.9:
            return (None, None)

        values.sort()
        tail = (1.0 - ci_level) / 2.0
        lo = values[int(len(values) * tail)]
        hi = values[int(len(values) * (1.0 - tail))]
        return (lo, hi)
