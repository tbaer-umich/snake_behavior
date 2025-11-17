#!/usr/bin/env python3
"""
utils.py

Utility functions for snake behavior classification:
  - break_into_chunks: split any sequence (e.g. DataFrame) into fixed-size slices
  - align_covariance_to_principal_axis: rotate covariance matrices to align their
    principal component (direction of maximum variance) with the x-axis
  - rotation_matrix_align_to_x: helper function to create rotation matrix using
    Rodrigues' formula
"""
import numpy as np

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

def align_covariance_to_principal_axis(cov_mat):
    """
    Rotate a covariance matrix so that its principal component
    (largest eigenvalue direction) points along the x-axis.

    Args:
        cov_mat: 3x3 covariance matrix (numpy array)

    Returns:
        Rotated 3x3 covariance matrix (numpy array)
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    # Find the principal component (eigenvector with largest eigenvalue)
    max_idx = np.argmax(eigenvalues)
    principal_component = eigenvectors[:, max_idx]

    # Create rotation matrix that aligns principal component with x-axis
    # We want the principal component to become [1, 0, 0]
    rotation = rotation_matrix_align_to_x(principal_component)

    # Rotate the covariance matrix: R * Cov * R^T
    rotated_cov = rotation @ cov_mat @ rotation.T

    return rotated_cov

def rotation_matrix_align_to_x(vector):
    """
    Create a rotation matrix that aligns the given vector with the x-axis [1,0,0].
    Uses Rodrigues' rotation formula.

    Args:
        vector: 3D vector to align with x-axis

    Returns:
        3x3 rotation matrix
    """
    # Normalize the input vector
    v = vector / np.linalg.norm(vector)

    # Target is the x-axis
    x_axis = np.array([1.0, 0.0, 0.0])

    # If vector is already aligned (or anti-aligned) with x-axis, return identity (or flip)
    dot_product = np.dot(v, x_axis)
    if np.abs(dot_product - 1.0) < 1e-10:
        return np.eye(3)
    if np.abs(dot_product + 1.0) < 1e-10:
        # Anti-aligned, flip around y-axis
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # Compute rotation axis (perpendicular to both vectors)
    rotation_axis = np.cross(v, x_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Compute rotation angle
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # Rodrigues' rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R
