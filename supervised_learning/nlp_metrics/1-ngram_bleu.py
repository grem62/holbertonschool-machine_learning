#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
    """

import numpy as np
from collections import Counter


def ngram_bleu(references, sentence, n):
    """calculates the n-gram BLEU score for a sentence"""
    def ngrams(sequence, n):
        """_summary_

        Args:
            sequence (_type_): _description_
            n (_type_): _description_

        Returns:
            _type_: _description_
        """
        return [' '.join(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

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
            ngram, 0))for ngram, count in sentence_ngrams_count.items()}

    precision = sum(clipped_counts.values()) / max(
        1, sum(sentence_ngrams_count.values()))

    ref_lengths = [len(ref) for ref in references]
    sen_length = len(sentence)

    closest_ref_length = min(
        ref_lengths, key=lambda ref_len: (
            abs(ref_len - sen_length), ref_len))

    if sen_length > closest_ref_length:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_length / sen_length)

    bleu_score = bp * precision
    return bleu_score
