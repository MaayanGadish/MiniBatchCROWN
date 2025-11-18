#!/usr/bin/env python3
import argparse, os, sys, torch, torch.nn as nn
import torch.fx as fx

def parse_bool(s:str)->bool: return str(s).lower() in {"1","true","yes","y","t"}

def _strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def _is_parameterized(mod: nn.Module) -> bool:
    # Count as a "layer" only if it has parameters (Conv/BN/Linear/etc).
    return sum(p.numel() for p in mod.parameters(recurse=False)) > 0

def build_prefix_graph(model: nn.Module, example: torch.Tensor, k: int) -> fx.GraphModule:
    """
    Trace the full model, then return a GraphModule that outputs the activation
    right after the first k parameterized modules. All necessary functional/method
    nodes up to that point are kept so the graph is executable.
    """
    model.eval()
    gm_full: fx.GraphModule = fx.symbolic_trace(model)

    kept_graph = fx.Graph()
    env = {}

    # Map from original node -> (cloned) new node
    param_mod_count = 0
    cut_at_node = None

    # Keep placeholder(s) first
    for node in gm_full.graph.nodes:
        if node.op == "placeholder":
            env[node] = kept_graph.placeholder(node.target)
        elif node.op == "get_attr":
            # parameters/buffers accessed via get_attr must be copied too
            env[node] = kept_graph.get_attr(node.target)
        else:
            # We'll process rest in next loop
            pass

    for node in gm_full.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue

        # Rebuild args/kwargs from env
        def map_arg(a):
            if isinstance(a, fx.Node): return env[a]
            if isinstance(a, (list, tuple)):
                return type(a)(map_arg(x) for x in a)
            if isinstance(a, dict):
                return {k: map_arg(v) for k, v in a.items()}
            return a

        new_node = None
        if node.op == "call_module":
            # Inspect the submodule to decide if this counts toward "k"
            submod = gm_full.get_submodule(node.target)
            new_node = kept_graph.call_module(node.target, map_arg(node.args), dict(node.kwargs))
            # Count only parameterized modules
            if _is_parameterized(submod):
                param_mod_count += 1
                cut_at_node = new_node
        elif node.op == "call_function":
            new_node = kept_graph.call_function(node.target, map_arg(node.args), dict(node.kwargs))
        elif node.op == "call_method":
            new_node = kept_graph.call_method(node.target, map_arg(node.args), dict(node.kwargs))
        elif node.op == "output":
            # we will set our own output later
            continue
        else:
            raise RuntimeError(f"Unsupported FX node op: {node.op}")

        env[node] = new_node

        # Stop once we’ve passed k parameterized modules.
        if param_mod_count >= k:
            cut_at_node = new_node
            break

    if cut_at_node is None:
        raise ValueError(
            "Could not find enough parameterized layers to cut at. "
            f"Requested k={k}, but model has fewer."
        )

    kept_graph.output(cut_at_node)
    gm_prefix = fx.GraphModule(gm_full, kept_graph)

    # Optional: remove unused submodules (PyTorch 2.1+ has this util)
    if hasattr(fx.graph_module, "delete_all_unused_submodules"):
        fx.graph_module.delete_all_unused_submodules(gm_prefix)

    return gm_prefix

def main():
    ap = argparse.ArgumentParser(description="Export first k layers of resnet2b to ONNX.")
    ap.add_argument("--weights", required=True, help="Path to resnet2b .pth/.pt")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--k", type=int, required=True, help="Number of parameterized layers to keep")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--strict-load", type=parse_bool, default=False)
    ap.add_argument("--dummy-batch", type=int, default=1)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    repo_root = os.path.abspath(".")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Import model constructor from the repo
    try:
        from complete_verifier.model_defs import resnet2b
    except Exception as e:
        raise RuntimeError(
            "Could not import resnet2b from complete_verifier/model_defs.py. "
            "Run this script from the alpha-beta-CROWN repo root."
        ) from e

    # Build model (try both signatures)
    try:
        model = resnet2b(num_classes=10)
    except TypeError:
        model = resnet2b()
    model.eval()

    # Load weights
    ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        sd = getattr(ckpt, "state_dict", None)
        if sd is None:
            raise RuntimeError("Unrecognized checkpoint format; expected state_dict or {'state_dict': ...}.")
        state_dict = sd
    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:   print("  e.g.,", missing[:5])
    if unexpected:print("  e.g.,", unexpected[:5])

    # Trace and cut after first k parameterized modules
    dummy = torch.randn(args.dummy_batch, 3, args.height, args.width)
    gm_prefix = build_prefix_graph(model, dummy, args.k)
    gm_prefix.eval()

    # Export ONNX of the prefix (its output is the cut activation, not logits)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        gm_prefix,
        dummy,
        args.out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[f"feat_k{args.k}"],
        dynamic_axes={"input": {0: "batch"}, f"feat_k{args.k}": {0: "batch"}},
    )
    print(f"[ok] wrote ONNX prefix (k={args.k}) -> {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
