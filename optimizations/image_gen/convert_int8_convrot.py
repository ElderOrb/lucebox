#!/usr/bin/env python3
"""MOSS-Image bf16 safetensors -> int8 convrot safetensors for stable-diffusion.cpp.

The four big linears of every transformer block (qkv, o, ffn_up_gate, ffn_down) are stored in the int8 tensorwise
convrot layout that ComfyUI and stable-diffusion.cpp read: each 256-wide input group of a weight row is rotated by the
normalized Hadamard matrix H4^(x4), the transform the runtime applies to the activations, and the row is quantized to
int8 with one scale per output row. Writes <module>.weight (I8), <module>.weight_scale (F32 [out]) and
<module>.comfy_quant (U8, the JSON config); every other tensor is copied unchanged. The rotation sums are exact in
float64 and rounded once to float32, so the output does not depend on the machine's BLAS.

Usage: convert_int8_convrot.py IN.safetensors OUT.safetensors   (needs numpy)
"""

import json
import struct
import sys

import numpy as np

GROUP = 256
CONFIG = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()
TARGETS = (".proj.qkv.weight", ".proj.o.weight", ".proj.ffn_up_gate.weight", ".proj.ffn_down.weight")


def hadamard_256():
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64) / 2
    rot = h4
    for _ in range(3):
        rot = np.kron(rot, h4)
    assert np.allclose(rot @ rot.T, np.eye(GROUP)) and np.allclose(rot, rot.T)
    return rot


def is_target(name, info):
    shape = info["shape"]
    return name.endswith(TARGETS) and info["dtype"] == "BF16" and len(shape) == 2 and shape[1] % GROUP == 0


def main(src, dst):
    rot = hadamard_256()
    with open(src, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    metadata = header.pop("__metadata__", None)

    plan, out_header, offset = [], {}, 0
    for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        if is_target(name, info):
            rows, cols = info["shape"]
            module = name[: -len(".weight")]
            for out_name, dtype, shape, size in (
                (name, "I8", [rows, cols], rows * cols),
                (module + ".weight_scale", "F32", [rows], rows * 4),
                (module + ".comfy_quant", "U8", [len(CONFIG)], len(CONFIG)),
            ):
                out_header[out_name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
                offset += size
            plan.append((info, True))
        else:
            begin, end = info["data_offsets"]
            out_header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
            offset += end - begin
            plan.append((info, False))
    if metadata is not None:
        out_header["__metadata__"] = metadata

    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    converted = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for info, convert in plan:
            begin, end = info["data_offsets"]
            fin.seek(data_start + begin)
            raw = fin.read(end - begin)
            if not convert:
                fout.write(raw)
                continue
            rows, cols = info["shape"]
            w = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            w_rot = (w.astype(np.float64).reshape(rows, cols // GROUP, GROUP) @ rot).astype(np.float32).reshape(rows, cols)
            scale = np.abs(w_rot).max(axis=1) / np.float32(127.0)
            safe = np.where(scale == 0.0, np.float32(1.0), scale)
            q = np.clip(np.rint(w_rot / safe[:, None]), -127, 127).astype(np.int8)
            fout.write(q.tobytes())
            fout.write(scale.astype(np.float32).tobytes())
            fout.write(CONFIG)
            converted += 1
    print(f"converted {converted} linears to int8 convrot, copied {len(plan) - converted} tensors, {offset / 1e9:.2f} GB of data")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(*sys.argv[1:])
