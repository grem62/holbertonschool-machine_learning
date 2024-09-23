#!/usr/bin/env python3

from collections import Counter
import numpy as np
import tensorflow as tf


def uni_bleu(references, sentence):
    """calculates the unigram BLEU score for a sentence"""

    # Count the words in the sentence
    sentence_counts = Counter(sentence)

    # Count the words in the references
    max_counts = {}
    for reference in references:
        reference_counts = Counter(reference)
        for word in reference_counts:
            if word in max_counts:
                max_counts[word] = max(
                    max_counts[word], reference_counts[word])
            else:
                max_counts[word] = reference_counts[word]

    # Calculate the clipped counts
    clipped_counts = {word: min(count, max_counts.get(word, 0))
                      for word, count in sentence_counts.items()}

    # Calculate precision
    precision = sum(clipped_counts.values()) / len(sentence)

    # Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_length = min(ref_lengths, key=lambda ref_len:
                             (abs(ref_len - len(sentence)), ref_len))
    if len(sentence) > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / len(sentence))

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision
    return bleu_score
