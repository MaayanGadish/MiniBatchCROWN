from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_dataset("scikit-learn/iris")
iris.set_format("pandas")
df = iris["train"][:]

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].to_numpy()
y = df["Species"].to_numpy()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# pick first test sample
x0 = X_test[0]
label = y_test[0]
print("Center x0:", x0)
print("Label:", label)

# Map label names to indices
label_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

label_idx = label_map[label]

# Parameters
epsilon = 0.005
output_file = f"iris_eps{epsilon:.3f}_label{label_idx}.vnnlib"

# Generate VNNLIB file
with open(output_file, "w") as f:
    f.write("; 4 inputs and 3 outputs (Iris MLP)\n\n")

    # Declare input variables
    for i in range(4):
        f.write(f"(declare-const X_{i} Real)\n")

    f.write("\n")

    # Declare output variables
    for i in range(3):
        f.write(f"(declare-const Y_{i} Real)\n")

    f.write("\n; Input bounds with epsilon perturbation\n")

    # Add input bounds with epsilon
    for i in range(4):
        lower = x0[i] - epsilon
        upper = x0[i] + epsilon
        f.write(f"(assert (>= X_{i} {lower:.3f}))\n")
        f.write(f"(assert (<= X_{i} {upper:.3f}))\n")

    f.write(f"\n; Robustness property: checking if adversarial example exists\n")
    f.write(f"; UNSAT = network is robust (maintains label-{label_idx} for all inputs)\n")
    f.write(f"; SAT = adversarial example found (network is not robust)\n")

    # Assert that at least one incorrect class has output >= true class
    # Each disjunct must be wrapped in (and ...) even if single condition
    f.write("(assert (or\n")
    for i in range(3):
        if i != label_idx:
            f.write(f"            (and (>= Y_{i} Y_{label_idx}))\n")
    f.write("        )\n")
    f.write(")\n")

print(f"VNNLIB file created: {output_file}")
