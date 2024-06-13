# File: rotation_matrix.py
import torch

def get_rotation_matrix(dim, context_size, period):
    """
    Generates a rotation matrix for positional encoding.

    Args:
        dim (int): The dimension of the space where the rotation is applied.
        context_size (int): The size of the context or sequence length.
        period (float): The period for the rotation which influences the frequency of rotation.

    Returns:
        torch.Tensor: A rotation matrix of shape (context_size, dim).
    """
    # Generate frequency for each dimension
    freqs = torch.pow(period, torch.arange(0, dim, 2).float() / dim)
    # Generate token indices for each position in the context
    positions = torch.arange(context_size).unsqueeze(1)
    # Compute the angle for rotation
    angles = positions * freqs.unsqueeze(0)
    # Calculate sine and cosine components
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    # Interleave sin and cos to form the rotation matrix
    rotation_matrix = torch.empty(context_size, dim)
    rotation_matrix[:, 0::2] = cos
    rotation_matrix[:, 1::2] = sin

    return rotation_matrix

if __name__ == "__main__":
    # Example usage of the get_rotation_matrix function
    dim = 10  # Embedding dimension
    context_size = 50  # Sequence length
    period = 10000.0  # A typical period for rotations
    rotation_matrix = get_rotation_matrix(dim, context_size, period)
    print("Rotation Matrix Shape:", rotation_matrix.shape)
    print("Rotation Matrix:", rotation_matrix)