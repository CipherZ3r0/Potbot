"""
Evaluation utility metrics for retrieval and generation without heavy external dependencies.
"""

import math
from collections import Counter
from typing import List, Set


def precision_at_k(actual: List[str], predicted: List[str], k: int) -> float:
    if not predicted or not actual:
        return 0.0
    pred_k = predicted[:k]
    hits = sum(1 for p in pred_k if p in actual)
    return hits / k


def recall_at_k(actual: List[str], predicted: List[str], k: int) -> float:
    if not predicted or not actual:
        return 0.0
    pred_k = predicted[:k]
    hits = sum(1 for p in pred_k if p in actual)
    return hits / len(actual)


def dcg_at_k(actual: List[str], predicted: List[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def ndcg_at_k(actual: List[str], predicted: List[str], k: int) -> float:
    dcg = dcg_at_k(actual, predicted, k)
    idcg = dcg_at_k(actual, actual, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(actual: List[str], predicted: List[str]) -> float:
    if not actual:
        return 0.0
    ap = 0.0
    hits = 0
    for i, p in enumerate(predicted):
        if p in actual:
            hits += 1
            ap += hits / (i + 1)
    return ap / len(actual)


def mean_average_precision(actuals: List[List[str]], predicteds: List[List[str]]) -> float:
    if not actuals:
        return 0.0
    return sum(average_precision(a, p) for a, p in zip(actuals, predicteds)) / len(actuals)


def token_overlap_f1(expected_answer: str, actual_answer: str) -> float:
    """Compute F1 score based on token overlap (ignoring punctuation)."""
    import re
    
    def get_tokens(text: str) -> List[str]:
        return [t for t in re.split(r'\W+', text.lower()) if t]

    expected_tokens = get_tokens(expected_answer)
    actual_tokens = get_tokens(actual_answer)

    if not expected_tokens or not actual_tokens:
        return 0.0

    common = Counter(expected_tokens) & Counter(actual_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(actual_tokens)
    recall = 1.0 * num_same / len(expected_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def simple_rouge_l(expected_answer: str, actual_answer: str) -> float:
    """Compute a simplified ROUGE-L (Longest Common Subsequence) F1 score."""
    import re
    
    def get_tokens(text: str) -> List[str]:
        return [t for t in re.split(r'\W+', text.lower()) if t]

    expected_tokens = get_tokens(expected_answer)
    actual_tokens = get_tokens(actual_answer)

    if not expected_tokens or not actual_tokens:
        return 0.0

    # DP for LCS length
    n, m = len(expected_tokens), len(actual_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected_tokens[i - 1] == actual_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    return (2 * precision * recall) / (precision + recall)
