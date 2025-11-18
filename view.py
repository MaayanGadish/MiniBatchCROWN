import torch
import torch.nn as nn

# Load the checkpoint first to inspect it
checkpoint = torch.load("first_k_layers.pth", map_location="cpu", weights_only=False)

# Extract the state dict (it's nested under "state_dict" key)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    print(
        f"Checkpoint info - Test Acc: {checkpoint.get('test_acc', 'N/A')}, Test Robust Acc: {checkpoint.get('test_robust_acc', 'N/A')}\n"
    )
else:
    state_dict = checkpoint


# Create a simple wrapper to hold the state dict
class ModelWrapper(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        # Create parameters from the state dict
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                self.register_buffer(name.replace(".", "_"), param)


# Use the state dict directly instead of trying to load into a model
model_state = state_dict

# Print model architecture (layer names)
print("=" * 80)
print("MODEL ARCHITECTURE (Layer Names)")
print("=" * 80)
layer_groups = {}
for name in model_state.keys():
    layer_name = name.split(".")[0]
    if layer_name not in layer_groups:
        layer_groups[layer_name] = []
    layer_groups[layer_name].append(name)

for layer_name, params in sorted(layer_groups.items()):
    print(f"\n{layer_name}:")
    for param in params:
        print(f"  - {param}")

# Print all parameters with their shapes
print("\n" + "=" * 80)
print("MODEL PARAMETERS")
print("=" * 80)
for name, param in model_state.items():
    if isinstance(param, torch.Tensor):
        print(f"{name:50s} | Shape: {str(param.shape):30s} | dtype: {param.dtype}")

print()
total_params = sum(
    p.numel() for p in model_state.values() if isinstance(p, torch.Tensor)
)
print(f"Total parameters: {total_params:,}")

# Print actual weight values for all layers
print("\n" + "=" * 80)
print("ACTUAL WEIGHT VALUES FOR ALL LAYERS")
print("=" * 80)

for name, param in model_state.items():
    if isinstance(param, torch.Tensor):
        print(f"\n{name}:")
        print(f"  Shape: {param.shape}")
        print(f"  Values:")

        # Set print options for better readability
        torch.set_printoptions(
            precision=6, threshold=10000, linewidth=100, sci_mode=False
        )

        # Print the actual tensor values
        print(param)
        print()
