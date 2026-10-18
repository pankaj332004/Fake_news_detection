"""
Text chunking module for passage segmentation.
Splits documents into coherent passages aligned on sentence boundaries with configurable overlap.
"""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using regex boundary matching.
    Handles abbreviations reasonably well.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # Match sentence terminators (. ! ?) followed by whitespace and capital letter or end of string
    sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
    raw_sentences = sentence_end.split(clean_text)

    # Filter out empty entries
    return [s.strip() for s in raw_sentences if s.strip()]


def chunk_text(
    text: str,
    max_chunk_chars: int = 400,
    overlap_chars: int = 60,
    min_chunk_chars: int = 80,
) -> List[str]:
    """
    Splits text into multi-sentence passages preserving semantic coherence.

    Args:
        text: Input long text.
        max_chunk_chars: Target maximum character length for a single chunk.
        overlap_chars: Target character overlap between consecutive chunks.
        min_chunk_chars: Discard or merge trailing chunks smaller than this size.

    Returns:
        List of passage strings.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_length = 0

    for sent in sentences:
        sent_len = len(sent)

        # If adding this sentence exceeds max_chunk_chars and we already have content
        if current_length + sent_len > max_chunk_chars and current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append(chunk_str)

            # Keep tail sentences for overlap
            overlap_sentences = []
            overlap_len = 0
            for prev_sent in reversed(current_sentences):
                if overlap_len + len(prev_sent) <= overlap_chars:
                    overlap_sentences.insert(0, prev_sent)
                    overlap_len += len(prev_sent)
                else:
                    break

            current_sentences = overlap_sentences + [sent]
            current_length = sum(len(s) for s in current_sentences) + len(current_sentences) - 1
        else:
            current_sentences.append(sent)
            current_length += sent_len + 1

    # Append remaining
    if current_sentences:
        last_chunk = " ".join(current_sentences)
        if chunks and len(last_chunk) < min_chunk_chars:
            # Merge with previous chunk if too short
            chunks[-1] = chunks[-1] + " " + last_chunk
        else:
            chunks.append(last_chunk)

    return chunks
