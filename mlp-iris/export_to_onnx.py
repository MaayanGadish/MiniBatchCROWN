import torch, torch.nn as nn
from datasets import load_dataset
from huggingface_hub import PyTorchModelHubMixin

# --- 1) Define the original MLP exactly like the model card ---
class MLP(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
    def forward(self, x):
        act = torch.relu(self.fc1(x))
        return self.fc2(act)

# --- 2) Wrap it with a normalization layer so ONNX includes preprocessing ---
class WithNorm(nn.Module):
    def __init__(self, core: nn.Module, means, stds):
        super().__init__()
        self.core = core
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("stds", torch.tensor(stds, dtype=torch.float32))
    def forward(self, x):
        x = (x - self.means) / self.stds
        return self.core(x)

# --- 3) Load training stats (means/stds) from the Iris dataset split used on the card ---
iris = load_dataset("scikit-learn/iris"); iris.set_format("pandas")
df = iris["train"][:]
X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]].to_numpy()
import numpy as np
# Match the model card’s normalization: compute from the training split
# (using a deterministic split here for reproducibility)
from sklearn.model_selection import train_test_split
_, X_test = None, None  # unused, but we replicate the split logic
X_train_full, X_test, y_train_full, y_test = train_test_split(X, df["Species"].to_numpy(), test_size=0.1, stratify=df["Species"].to_numpy(), random_state=42)
X_train, X_valid, _, _ = train_test_split(X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42)
means = X_train.mean(axis=0).astype("float32")
stds  = X_train.std(axis=0).astype("float32")

# --- 4) Load HF weights and wrap with normalization ---
base = MLP().from_pretrained("sadhaklal/mlp-iris").eval()
model = WithNorm(base, means, stds).eval()

# --- 5) Export to ONNX (opset 13+, fixed input size 4) ---
dummy = torch.zeros(1, 4, dtype=torch.float32)
torch.onnx.export(
    model, dummy, "../mlp_iris.onnx",
    input_names=["X"], output_names=["Y"],
    dynamic_axes=None, opset_version=13, do_constant_folding=True
)
print("Exported to models/mlp_iris.onnx")
