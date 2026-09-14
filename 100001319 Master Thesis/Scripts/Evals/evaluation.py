"""Stage 5, Step 1: empirical error and recall on a held-out test split,
given a frozen lambda_hat from Stage 4 calibration.

Pure metric computation, no I/O -- shared by scripts/evaluate.py (real data,
a later step) and tests/test_evaluation.py (synthetic, known-truth data).
Per docs/stage_5.md: "Empirical error vs. nominal alpha, on the held-out test
split (never touched during calibration)" is the primary figure; this module
computes the two numbers (error, recall) that figure plots, for one
(lambda_hat, loss_policy) pair at a time. The figures themselves, the
fixed-threshold and self-consistency-alone baselines, and per-scenario
breakdown are later steps -- not built here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Same wrong-label policy definitions as src/calibrate.py's ltt_calibrate /
# crc_calibrate. Duplicated rather than imported: evaluation must be callable
# against any lambda_hat (including ones NOT produced by this run of
# calibration, e.g. a fixed-threshold baseline in a later step) without
# pulling in the LTT grid-search machinery those functions carry. A two-line
# policy definition is cheaper to keep in sync by hand than to entangle
# evaluation with calibration's internals for this alone. If the policy
# definitions ever diverge, tests/test_evaluation.py's
# TestWrongLabelsAgreeWithCalibrate catches it.
_PRIMARY_WRONG_LABELS = {"hallucinated", "misweighted", "false precision"}
_SECONDARY_WRONG_LABELS = {"hallucinated"}


def _wrong_labels(loss_policy: str) -> set[str]:
    if loss_policy == "primary":
        return _PRIMARY_WRONG_LABELS
    if loss_policy == "secondary":
        return _SECONDARY_WRONG_LABELS
    raise ValueError(f"loss_policy must be 'primary' or 'secondary', got {loss_policy!r}")


@dataclass
class EvaluationResult:
    """Empirical error and recall for one (lambda_hat, loss_policy) pair on
    one test split."""

    lambda_hat: float
    loss_policy: str
    n_test: int
    n_accepted: int
    n_wrong_accepted: int
    n_grounded: int
    n_grounded_accepted: int
    error: float   # L(lambda_hat) on test: wrong-accepted / max(accepted, 1)
    recall: float  # grounded-accepted / max(grounded-total, 1)


def evaluate(
    rhos: Sequence[float],
    labels: Sequence[str],
    lambda_hat: float,
    loss_policy: str,
) -> EvaluationResult:
    """Compute L(lambda_hat) (empirical error) and grounded-recall on a test
    split.

    Accepted set A_lambda = {s : rho(s) >= lambda_hat} -- the SAME acceptance
    rule Stage 4 calibrated against (src/calibrate.py's ltt_calibrate uses
    `r >= lam`); evaluation must use the identical rule or the guarantee
    Stage 4 produced is not the one being checked here.

    error = |{s in A_lambda : wrong}| / max(|A_lambda|, 1)
        CLAUDE.md's own loss definition, computed on the TEST split rather
        than the calibration split Stage 4 used. "wrong" is defined by
        `loss_policy`, exactly matching src/calibrate.py's wrong_labels sets.

    recall = |{s in A_lambda : grounded}| / max(|{s : grounded}|, 1)
        The docs/stage_5.md "usefulness" half: of all grounded sentences in
        the test split, what fraction survive the filter. This is recall of
        grounded content, NOT precision of the accepted set (fraction of
        A_lambda that is grounded) -- a filter that accepts everything has
        perfect recall by this definition regardless of how much wrong
        content also gets through, which is exactly why docs/stage_5.md
        requires reporting error and recall together, never one alone.
    """
    if len(rhos) != len(labels):
        raise ValueError("rhos and labels must have the same length")
    wrong_labels = _wrong_labels(loss_policy)

    n_test = len(rhos)
    accepted = [i for i, r in enumerate(rhos) if r >= lambda_hat]
    n_accepted = len(accepted)
    n_wrong_accepted = sum(1 for i in accepted if labels[i] in wrong_labels)

    grounded_idx = [i for i, lbl in enumerate(labels) if lbl == "grounded"]
    n_grounded = len(grounded_idx)
    n_grounded_accepted = sum(1 for i in grounded_idx if rhos[i] >= lambda_hat)

    return EvaluationResult(
        lambda_hat=lambda_hat,
        loss_policy=loss_policy,
        n_test=n_test,
        n_accepted=n_accepted,
        n_wrong_accepted=n_wrong_accepted,
        n_grounded=n_grounded,
        n_grounded_accepted=n_grounded_accepted,
        error=n_wrong_accepted / max(n_accepted, 1),
        recall=n_grounded_accepted / max(n_grounded, 1),
    )
