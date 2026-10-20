import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    res = torch.sigmoid(torch.tensor(z, dtype=torch.float32))
    return round(float(res), 4)
