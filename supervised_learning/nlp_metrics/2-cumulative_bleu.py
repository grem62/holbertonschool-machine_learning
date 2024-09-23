#!/usr/bin/env python3
"""
Calculates the cumulative n-gram BLEU score for a sentence.
"""
import numpy as np
from collections import Counter


def ngram_precision(references, sentence, n):
    """
    Calculates the precision for n-grams.
    """

    def ngrams(sequence, n):
        """_summary_

        Args:
            sequence (_type_): _description_
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [tuple(
            sequence[i:i+n]) for i in range(len(sequence)-n+1)]

    sentence_ngrams = ngrams(sentence, n)
    sentence_ngrams_count = Counter(sentence_ngrams)

    max_counts = {}
    for reference in references:
        reference_ngrams = ngrams(reference, n)
        reference_ngrams_count = Counter(reference_ngrams)
        for ngram in sentence_ngrams_count:
            if ngram in max_counts:
                max_counts[ngram] = max(
                    max_counts[ngram], reference_ngrams_count[ngram])
            else:
                max_counts[ngram] = reference_ngrams_count[ngram]

    clipped_counts = {ngram: min(
        count, max_counts.get(
            ngram, 0)) for ngram, count in sentence_ngrams_count.items()}
    precision = sum(clipped_counts.values()) / max(1, sum(
        sentence_ngrams_count.values()))

    return precision


def brevity_penalty(references, sentence):
    """
    Calculates the brevity penalty for a sentence.
    """
    ref_lens = [len(ref) for ref in references]
    sen_len = len(sentence)

    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (
            abs(ref_len - sen_len), ref_len))

    if sen_len > closest_ref_len:
        return 1
    else:
        return np.exp(1 - closest_ref_len / sen_len)


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    """
    weights = [1/n] * n
    p_n = [ngram_precision(references, sentence, i) for i in range(1, n+1)]
    s = np.log(p_n)
    score = np.exp(np.dot(weights, s))
    bp = brevity_penalty(references, sentence)
    return bp * score
