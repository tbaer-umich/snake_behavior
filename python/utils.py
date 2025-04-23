#!/usr/bin/env python3
"""
utils.py

Helper function for straightforward chunk splitting:
  - break_into_chunks: split any sequence (e.g. DataFrame) into fixed-size slices
"""

def break_into_chunks(df, chunk_size):
    """
    Splits `df` (or any sequence with `len` and support for slicing) into consecutive
    chunks of at most `chunk_size` rows or items.

    Parameters:
      df: sequence-like (e.g. pandas.DataFrame, numpy array, list)
      chunk_size (int): maximal number of entries per chunk

    Returns:
      List[Tuple[int,int]]: list of (start_index, end_index) pairs, end exclusive
    """
    n = len(df)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((start, end))
    return chunks

