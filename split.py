import torch
from collections import OrderedDict


def save_first_k_layers(pth_path, k, output_path="first_k_layers.pth"):
    """
    Loads a PyTorch .pth file and saves a new .pth file
    containing only the first k layers' parameters.

    Args:
        pth_path (str): Path to the original .pth file.
        k (int): Number of layers to keep.
        output_path (str): Path to save the new .pth file.

    Returns:
        str: Path to the saved file.
    """
    # Load the checkpoint
    checkpoint = torch.load(pth_path, map_location="cpu")

    # If it's a checkpoint with a state dict inside
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        # assume it's directly a state_dict
        state_dict = checkpoint
    else:
        raise ValueError(
            "The .pth file format is not recognized (not a dict or missing state_dict)."
        )

    # Keep only first k parameter tensors
    new_state_dict = OrderedDict(list(state_dict.items())[:k])

    # Save to a new file
    torch.save(new_state_dict, output_path)

    print(f"Saved first {k} layers to {output_path}")
    return output_path


if __name__ == "__main__":
    save_first_k_layers("resnet2b.pth", 6)
