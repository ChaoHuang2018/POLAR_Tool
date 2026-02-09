#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert POLAR txt-format MLP to JSON schema (for POLAR new loader) and/or ONNX.

- Input txt format (as used in POLAR):
    line 1:   num_of_inputs
    line 2:   num_of_outputs
    line 3:   num_of_hidden_layers
    next H lines:   hidden layer sizes (H = num_of_hidden_layers)
    next H+1 lines: activations for all hidden layers + output layer
    then for each layer L0..LH (H hidden + output):
        weights row by row, and a bias at end of each row
        (first layer: [hidden0 x in], then bias[hidden0];
         next: [hidden1 x hidden0], ..., last: [out x prev])
    finally:
        offset
        scale_factor

- JSON schema (output):
{
  "type": "mlp",
  "in_dim": 4,
  "out_dim": 2,
  "layers": [
    {"W": [[...],[...]], "b":[...], "act":"relu"},
    ...
  ],
  "offset": 0.0,
  "scale": 1.0,
  "name": "optional"
}

- ONNX export:
    input: [in] (1D), then Unsqueeze -> [1,in]
    (optional) scale/offset preprocessing: x_scaled = x*scale + offset
    per layer: Gemm(A=x, B=W, C=b, transB=1) -> [1,out]
               activation (Relu/Tanh/Sigmoid/Identity)
    final Squeeze -> [out]

NOTE:
- Activations mapping:
    txt: "ReLU" -> onnx "Relu" / json "relu"
         "tanh" -> "Tanh" / "tanh"
         "sigmoid" -> "Sigmoid" / "sigmoid"
         "Affine" (or others) -> Identity / "linear"
- ONNX opset: 13
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple, Dict, Any

import numpy as np

try:
    import onnx
    from onnx import helper as oh, TensorProto, numpy_helper
except Exception:
    onnx = None


# ---------- Parsing POLAR txt ----------

def parse_polar_txt(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() != ""]

    it = iter(lines)

    def next_int():
        return int(next(it))

    def next_float():
        return float(next(it))

    in_dim = next_int()
    out_dim = next_int()
    num_hidden = next_int()

    hidden_sizes = []
    for _ in range(num_hidden):
        hidden_sizes.append(next_int())

    # activations for hidden + output (length = num_hidden + 1)
    activations = []
    for _ in range(num_hidden + 1):
        activations.append(next(it))

    # layers sizes: [h0, h1, ..., out]
    layer_out_dims = hidden_sizes + [out_dim]
    layer_in_dims = [in_dim] + hidden_sizes

    layers: List[Dict[str, Any]] = []
    for L, (m, n) in enumerate(zip(layer_out_dims, layer_in_dims)):
        # read W (m x n) row-major, then bias (m)
        W = np.zeros((m, n), dtype=np.float64)
        b = np.zeros((m,), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                W[i, j] = float(next(it))
            b[i] = float(next(it))
        act_txt = activations[L]
        layers.append({"W": W, "b": b, "act_txt": act_txt})

    offset = float(next(it))
    scale = float(next(it))

    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "hidden": hidden_sizes,
        "layers": layers,
        "offset": offset,
        "scale": scale,
    }


# ---------- Activation mapping ----------

def map_act_txt_to_json(act: str) -> str:
    a = act.strip()
    al = a.lower()
    if al == "relu" or a == "ReLU":
        return "relu"
    if al == "tanh":
        return "tanh"
    if al == "sigmoid":
        return "sigmoid"
    # "Affine" or anything else -> linear
    return "linear"


def map_act_txt_to_onnx(act: str) -> str:
    a = act.strip()
    al = a.lower()
    if al == "relu" or a == "ReLU":
        return "Relu"
    if al == "tanh":
        return "Tanh"
    if al == "sigmoid":
        return "Sigmoid"
    # "Affine" or anything else -> Identity
    return "Identity"


# ---------- Export JSON ----------

def export_json(parsed: Dict[str, Any], out_path: str, name: str = None):
    j = {
        "type": "mlp",
        "in_dim": parsed["in_dim"],
        "out_dim": parsed["out_dim"],
        "layers": [],
        "offset": parsed["offset"],
        "scale": parsed["scale"],
    }
    if name:
        j["name"] = name

    for L in parsed["layers"]:
        W = L["W"].tolist()
        b = L["b"].tolist()
        act_json = map_act_txt_to_json(L["act_txt"])
        j["layers"].append({"W": W, "b": b, "act": act_json})

    with open(out_path, "w") as f:
        json.dump(j, f, indent=2)
    print(f"[OK] JSON saved to {out_path}")


# ---------- Export ONNX (Gemm + transB=1) ----------

def export_onnx(parsed, out_path, model_name="mlp_from_polar_txt", opset=13):
    if onnx is None:
        raise RuntimeError("onnx package not installed. `pip install onnx`")

    in_dim  = parsed["in_dim"]
    out_dim = parsed["out_dim"]
    layers  = parsed["layers"]
    offset  = float(parsed["offset"])
    scale   = float(parsed["scale"])

    nodes, inits = [], []

    # graph IO
    inp = oh.make_tensor_value_info("input",  TensorProto.DOUBLE, [in_dim])
    out = oh.make_tensor_value_info("output", TensorProto.DOUBLE, [out_dim])

    curr = "input"

    # 取消输入端 Mul/Add；只做形状对齐
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes_unsq0"))
    nodes.append(oh.make_node("Unsqueeze", inputs=[curr, "axes_unsq0"], outputs=["x2d"], name="unsqueeze_in"))
    prev = "x2d"

    last_out_dim = None
    for li, L in enumerate(layers):
        W = L["W"].astype(np.float64)  # [out,in]
        b = L["b"].astype(np.float64)  # [out]
        act_onnx = map_act_txt_to_onnx(L["act_txt"])

        W_name, b_name = f"W_{li}", f"b_{li}"
        inits.append(numpy_helper.from_array(W, name=W_name))
        inits.append(numpy_helper.from_array(b, name=b_name))

        gemm_out = f"lin_{li}"
        nodes.append(oh.make_node(
            "Gemm",
            inputs=[prev, W_name, b_name],
            outputs=[gemm_out],
            name=f"gemm_{li}",
            transB=1
        ))
        last_out_dim = W.shape[0]
        prev = gemm_out

        if act_onnx != "Identity":
            act_out = f"act_{li}"
            nodes.append(oh.make_node(act_onnx, inputs=[prev], outputs=[act_out], name=f"act_{li}_{act_onnx}"))
            prev = act_out

    # === 输出端仿射： y' = scale * (y - offset) ===
    if (not math.isclose(scale, 1.0)) or (not math.isclose(offset, 0.0)):
        # 向量 [out_dim]，每个元素=offset
        offset_vec = np.full((last_out_dim,), offset, dtype=np.float64)
        inits.append(numpy_helper.from_array(offset_vec, name="offset_vec"))
        nodes.append(oh.make_node("Sub", inputs=[prev, "offset_vec"], outputs=["y_minus_off"], name="sub_offset"))

        inits.append(numpy_helper.from_array(np.array(scale, dtype=np.float64), name="scale_const"))
        nodes.append(oh.make_node("Mul", inputs=["y_minus_off", "scale_const"], outputs=["y_scaled"], name="mul_scale"))
        prev = "y_scaled"

    # Squeeze -> [out]
    inits.append(numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes_sq0"))
    nodes.append(oh.make_node("Squeeze", inputs=[prev, "axes_sq0"], outputs=["output"], name="squeeze_out"))

    g = oh.make_graph(nodes=nodes, name="mlp_graph", inputs=[inp], outputs=[out], initializer=inits)
    opset_imports = [oh.make_operatorsetid("", opset)]
    model = oh.make_model(g, opset_imports=opset_imports, producer_name="txt2net")
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"[OK] ONNX saved to {out_path} (opset {opset})")


# ---------- Numpy forward (for spot-check) ----------

def np_forward(parsed: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x * float(parsed["scale"]) + float(parsed["offset"])

    def act(a: np.ndarray, name: str):
        name = name.strip()
        low = name.lower()
        if low == "relu" or name == "ReLU":
            return np.maximum(a, 0.0)
        if low == "tanh":
            return np.tanh(a)
        if low == "sigmoid":
            return 1.0 / (1.0 + np.exp(-a))
        # Affine / linear
        return a

    h = x
    for L in parsed["layers"]:
        W = L["W"]
        b = L["b"]
        a = W @ h + b
        h = act(a, L["act_txt"])
    return h


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Convert POLAR txt MLP to JSON and/or ONNX.")
    ap.add_argument("--input", "-i", required=True, help="Path to POLAR .txt model")
    ap.add_argument("--to", nargs="+", choices=["json", "onnx"], required=True, help="Targets to export")
    ap.add_argument("--output", "-o", nargs="+", required=True, help="Output path(s) in the same order as --to")
    ap.add_argument("--name", default=None, help="Optional model name (JSON only)")
    ap.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    ap.add_argument("--check", type=int, default=0, help="Spot-check N random inputs for JSON forward equality")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for spot-check")
    args = ap.parse_args()

    if len(args.to) != len(args.output):
        print("ERROR: --to and --output must have the same length", file=sys.stderr)
        sys.exit(1)

    parsed = parse_polar_txt(args.input)

    # Exports
    for kind, outp in zip(args.to, args.output):
        if kind == "json":
            export_json(parsed, outp, args.name)
        elif kind == "onnx":
            export_onnx(parsed, outp, model_name=args.name or "mlp_from_polar_txt", opset=args.opset)

    # Optional quick consistency check (txt vs JSON forward)
    if args.check > 0:
        rng = np.random.default_rng(args.seed)
        in_dim = parsed["in_dim"]
        ok = True
        for t in range(args.check):
            x = rng.standard_normal(in_dim)
            y_txt = np_forward(parsed, x)

            # reload from JSON file we just wrote (if any)
            json_outs = [p for k, p in zip(args.to, args.output) if k == "json"]
            if not json_outs:
                print("[WARN] --check skipped (no JSON was exported)")
                break
            with open(json_outs[0], "r") as f:
                j = json.load(f)
            # build numpy forward from JSON
            layers = []
            for L in j["layers"]:
                W = np.array(L["W"], dtype=np.float64)
                b = np.array(L["b"], dtype=np.float64)
                act = L.get("act", "linear")
                layers.append({"W": W, "b": b, "act_txt": act})
            parsed_json_like = {
                "in_dim": j["in_dim"],
                "out_dim": j["out_dim"],
                "layers": layers,
                "offset": j.get("offset", 0.0),
                "scale": j.get("scale", 1.0),
            }
            y_json = np_forward(parsed_json_like, x)
            if not np.allclose(y_txt, y_json, rtol=1e-9, atol=1e-9):
                ok = False
                print("[CHECK] mismatch at sample", t, "\n x =", x, "\n txt =", y_txt, "\n json=", y_json)
                break
        if ok:
            print(f"[OK] spot-check passed for {args.check} random inputs (txt vs JSON).")


if __name__ == "__main__":
    main()
