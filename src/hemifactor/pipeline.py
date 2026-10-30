"""End-to-end phase runners (0–5) producing JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hemifactor.calibrate.activations import (
    accumulate_cov,
    mlp_intermediate,
    sample_residual_gaussian,
)
from hemifactor.calibrate.distortion import activation_mse, distortion_vs_roundtrip
from hemifactor.config import artifacts_dir, load_config
from hemifactor.eval.roofline import compare_buses
from hemifactor.eval.traffic import bytes_per_token, ratio_vs_baseline, residency_ok
from hemifactor.factorize.monarch import fit_monarch
from hemifactor.factorize.quantize_factors import quantize_hemifactor
from hemifactor.factorize.whitened import hemifactor_whitened
from hemifactor.gguf_io.inspect import summarize_gguf
from hemifactor.gguf_io.tensors import dequantize_named, iter_role, tensor_inventory
from hemifactor.kernels.numpy_fused import bench_fused
from hemifactor.spectrum.whiten import spectrum_report, whitened_singular_values


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=default))


def _jsonable_cov(stats: dict) -> dict:
    out = {k: v for k, v in stats.items() if k not in {"cov", "cov_sketch", "proj", "mean", "diag_var"}}
    if "participation_ratio" in stats:
        out["participation_ratio"] = float(stats["participation_ratio"])
    if "participation_ratio_diag" in stats:
        out["participation_ratio_diag"] = float(stats["participation_ratio_diag"])
    if "diag_var" in stats:
        dv = np.asarray(stats["diag_var"])
        out["diag_var_mean"] = float(dv.mean())
        out["diag_var_sum"] = float(dv.sum())
    return out


def phase0_freeze(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    gguf = cfg["model"].get("gguf_path")
    if not gguf:
        raise RuntimeError("model.gguf_path missing — set configs/default.yaml")
    import hashlib

    p = Path(gguf)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    digest = h.hexdigest()
    summary = summarize_gguf(p)
    inv = tensor_inventory(p)
    report = {
        "phase": 0,
        "status": "ok",
        "hard_constraints": "docs/hard-constraints.md",
        "note": "Exact codes in 6GB remains CLOSED; freeze is identity only.",
        "gguf_path": str(p.resolve()),
        "size_bytes": p.stat().st_size,
        "sha256": digest,
        "ollama_digest": cfg["model"].get("ollama_digest"),
        "summary_fields": {
            k: summary["fields"].get(k)
            for k in (
                "general.architecture",
                "general.base_model.0.name",
                "gemma4.block_count",
                "gemma4.embedding_length",
                "gemma4.context_length",
            )
            if k in summary["fields"]
        },
        "inventory_roles": inv["roles"],
        "n_tensors": inv["n_tensors"],
        "tensor_nbytes_sum": inv["tensor_nbytes_sum"],
    }
    _dump(art / "model_freeze.json", report)
    return report


def _heldout_texts(path: Path, n: int, seq_len: int) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # crude char chunks as stand-in tokens for activation sampling only
    chunks = []
    step = max(seq_len, 64)
    for i in range(0, min(len(raw) - step, n * step), step):
        chunks.append(raw[i : i + step])
        if len(chunks) >= n:
            break
    while len(chunks) < n:
        chunks.append(raw[:step] if raw else "heldout ")
    return chunks


def phase1_calibrate(cfg: dict[str, Any] | None = None, layers: list[int] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    gguf = cfg["model"]["gguf_path"]
    n_embd = int(cfg["model"]["n_embd"])
    n_ff = int(cfg["model"]["n_ff"])
    cal = cfg["calibration"]
    n_fit = int(cal["n_sequences_fit"])
    heldout_path = Path(cal["heldout_corpus_path"])
    rng = np.random.default_rng(0)

    # residual proxy: prefer embedding rows if affordable sample; else Gaussian
    method = "gaussian_residual_proxy"
    try:
        # sample by dequantizing is huge; use Gaussian scaled after a tiny probe
        residual = sample_residual_gaussian(n_fit, n_embd, rms=1.0, rng=rng)
        method = "gaussian_unit_rms_residual"
    except Exception:
        residual = sample_residual_gaussian(n_fit, n_embd, rms=1.0, rng=rng)

    if layers is None:
        layers = [int(x) for x in cfg.get("calibration", {}).get("probe_layers") or [0]]

    d_per_tensor: dict[str, Any] = {"method": method, "control_codec": "Q4_0", "tensors": {}}
    pr_per_layer: dict[str, Any] = {"method": method, "layers": {}}
    control = {"method": method, "control_codec": "Q4_0", "layers": {}}

    # held-out residual draws (forbidden to factor fit later — recorded separately)
    _ = _heldout_texts(heldout_path, int(cal["n_sequences_heldout"]), int(cal["seq_len"]))
    residual_held = sample_residual_gaussian(int(cal["n_sequences_heldout"]), n_embd, rms=1.0, rng=np.random.default_rng(1))

    for layer in layers:
        W_gate = dequantize_named(gguf, f"blk.{layer}.ffn_gate.weight")
        W_up = dequantize_named(gguf, f"blk.{layer}.ffn_up.weight")
        W_down = dequantize_named(gguf, f"blk.{layer}.ffn_down.weight")
        ff = mlp_intermediate(residual, W_gate, W_up)
        ff_h = mlp_intermediate(residual_held, W_gate, W_up)

        cov_ff = accumulate_cov(ff, sketch_dim=512, rng=rng)
        cov_res = accumulate_cov(residual, sketch_dim=None, rng=rng)

        # D control on down using held-out intermediates
        d_down = distortion_vs_roundtrip(W_down, ff_h, qtype_name="Q4_0")
        # also gate (residual-fed)
        d_gate = distortion_vs_roundtrip(W_gate, residual_held, qtype_name="Q4_0")

        ref_down = next(r for r in iter_role(gguf, "down") if r.layer == layer)
        d_per_tensor["tensors"][f"blk.{layer}.ffn_down.weight"] = {
            **d_down,
            "nbytes": ref_down.nbytes,
            "shape": list(W_down.shape),
        }
        d_per_tensor["tensors"][f"blk.{layer}.ffn_gate.weight"] = {
            **d_gate,
            "shape": list(W_gate.shape),
        }
        pr_per_layer["layers"][str(layer)] = {
            "residual": _jsonable_cov(cov_res),
            "ff_intermediate": _jsonable_cov(cov_ff),
        }
        control["layers"][str(layer)] = {
            "down_D_activation_mse": d_down["D_activation_mse"],
            "down_D_relative": d_down["D_relative"],
            "gate_D_activation_mse": d_gate["D_activation_mse"],
            "note": "Tensor-level Q6-dequant→Q4_0 control; not end-to-end NLL yet.",
        }

        # stash diag vars for phase 2 (compact)
        np.savez_compressed(
            art / f"cov_layer{layer}.npz",
            residual_diag=np.var(residual, axis=0, ddof=1),
            ff_diag=np.asarray(cov_ff["diag_var"]),
            residual_cov=cov_res["cov"],
            ff_samples_held=ff_h,
            residual_held=residual_held,
            W_down=W_down,  # large; only for selected layers
        )

    out = {
        "phase": 1,
        "status": "ok",
        "layers": layers,
        "activation_method": method,
        "heldout_corpus_path": str(heldout_path),
        "hard_constraints": "docs/hard-constraints.md",
        "caveat": (
            "Activations are MLP-surrogate samples (Gaussian residual → SiLU(gate)*up), "
            "not full-model hooked residuals. End-to-end NLL remains a separate bar."
        ),
    }
    _dump(art / "D_per_tensor.json", d_per_tensor)
    _dump(art / "participation_ratio_per_layer.json", pr_per_layer)
    _dump(art / "q6_to_q4_control.json", control)
    _dump(art / "phase1_status.json", out)
    return out


def phase2_spectrum(cfg: dict[str, Any] | None = None, layers: list[int] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    gguf = cfg["model"]["gguf_path"]
    ranks = list(cfg["spectrum"]["ranks_twosided"])
    factor_bits = float(cfg["prototype"]["factor_bits"])
    if layers is None:
        layers = [int(p.stem.replace("cov_layer", "")) for p in sorted(art.glob("cov_layer*.npz"))]
    spectra_dir = art / "spectra"
    spectra_dir.mkdir(exist_ok=True)

    decisions = []
    tables = []
    for layer in layers:
        pack = np.load(art / f"cov_layer{layer}.npz")
        W = pack["W_down"]
        diag = pack["ff_diag"]
        # down eats FF activations
        s = whitened_singular_values(W, diag_var=diag)
        ref = next(r for r in iter_role(gguf, "down") if r.layer == layer)
        # D floor from phase1
        d_doc = json.loads((art / "D_per_tensor.json").read_text())
        d_floor = float(d_doc["tensors"][f"blk.{layer}.ffn_down.weight"]["D_activation_mse"])
        rep = spectrum_report(
            s,
            d_floor=d_floor,
            candidate_ks=ranks,
            nbytes_baseline=ref.nbytes,
            out_dim=W.shape[0],
            in_dim=W.shape[1],
            factor_bits=factor_bits,
        )
        # save spectrum vector compactly
        np.save(spectra_dir / f"down_layer{layer}_sigma.npy", s)
        tables.append({"layer": layer, "role": "down", **{k: rep[k] for k in ("chosen_k", "licensed", "decision", "effective_rank_energy")}})
        decisions.append(rep["decision"])
        _dump(spectra_dir / f"down_layer{layer}.json", rep)

    # aggregate decision
    if all(d == "go_hemifactor" for d in decisions):
        aggregate = "go"
    elif any(d == "go_hemifactor" for d in decisions) or any(d == "borderline_traffic" for d in decisions):
        aggregate = "borderline"
    elif any(d.startswith("branch_b") for d in decisions):
        aggregate = "branch-B"
    else:
        aggregate = "stop"

    license_md = art / "rank_license.md"
    lines = [
        "# Rank license (Phase 2)",
        "",
        f"Aggregate decision: **{aggregate}**",
        "",
        "Per-layer down-projection (whitened by FF diag-var from Phase 1 surrogate μ):",
        "",
        "| layer | licensed | chosen_k | decision | eff_rank |",
        "|---|---|---|---|---|",
    ]
    for row in tables:
        lines.append(
            f"| {row['layer']} | {row['licensed']} | {row['chosen_k']} | {row['decision']} | {row['effective_rank_energy']:.1f} |"
        )
    lines += [
        "",
        "Activation method caveat: see `phase1_status.json`.",
        "Negative / branch-B outcomes are first-class results (hard-constraints E).",
        "",
    ]
    license_md.write_text("\n".join(lines))
    out = {"phase": 2, "status": "ok", "aggregate": aggregate, "tables": tables}
    _dump(art / "phase2_status.json", out)
    return out


def phase3_down_prototype(cfg: dict[str, Any] | None = None, layer: int | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    bits = int(cfg["prototype"]["factor_bits"])
    ranks = list(cfg["spectrum"]["ranks_twosided"])
    if layer is None:
        # pick first licensed layer or layer 0
        p2 = json.loads((art / "phase2_status.json").read_text())
        licensed = [t for t in p2["tables"] if t.get("licensed")]
        layer = int(licensed[0]["layer"]) if licensed else int(p2["tables"][0]["layer"])

    pack = np.load(art / f"cov_layer{layer}.npz")
    W = pack["W_down"]
    diag = pack["ff_diag"]
    x_h = pack["ff_samples_held"]
    d_doc = json.loads((art / "D_per_tensor.json").read_text())
    d_floor = float(d_doc["tensors"][f"blk.{layer}.ffn_down.weight"]["D_activation_mse"])
    spec = json.loads((art / "spectra" / f"down_layer{layer}.json").read_text())
    k = spec.get("chosen_k") or ranks[-1]
    ref = next(r for r in iter_role(cfg["model"]["gguf_path"], "down") if r.layer == layer)

    hf = hemifactor_whitened(W, k=int(k), role="down", layer=layer, diag_var=diag)
    hf_q, qmeta = quantize_hemifactor(hf, bits=bits)
    mse = activation_mse(W, hf_q.A @ hf_q.B, x_h)
    # Branch-B probe if not licensed — fit on a disjoint draw, eval on held-out x_h
    monarch_meta = None
    if not spec.get("licensed"):
        rng_m = np.random.default_rng(7)
        x_fit = rng_m.standard_normal((max(256, x_h.shape[0]), x_h.shape[1]), dtype=np.float64)
        # match held-out column scale
        scale = np.std(x_h, axis=0, ddof=1) + 1e-12
        x_fit = x_fit * scale
        mon = fit_monarch(W, x_fit, block=64, rng=rng_m)
        y = x_h @ W.T
        y_hat = mon.matvec_batch(x_h)
        monarch_meta = {
            "activation_mse": float(np.mean(np.sum((y - y_hat) ** 2, axis=1))),
            "nbytes_est": mon.nbytes_at_bits(bits),
            "block": mon.block,
            "fit_n": int(x_fit.shape[0]),
            "eval_n": int(x_h.shape[0]),
            "note": "Fit/eval split enforced; exact interpolation on eval is not allowed.",
        }

    factor_bytes = hf_q.nbytes_at_bytes_per_param(bits)
    byte_ratio = factor_bytes / max(ref.nbytes, 1)
    quality_pass = mse <= d_floor
    bytes_pass = byte_ratio <= 0.55
    status = "pass" if (quality_pass and bytes_pass) else "fail"
    # Do not "fix" by lowering bits elsewhere
    out_dir = art / "down_proj_prototype"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / f"layer{layer}_factors.npz", A=hf_q.A, B=hf_q.B)
    result = {
        "phase": 3,
        "status": status,
        "layer": layer,
        "k": int(k),
        "factor_bits": bits,
        "activation_mse": mse,
        "d_floor": d_floor,
        "quality_pass": quality_pass,
        "factor_nbytes_est": factor_bytes,
        "baseline_nbytes": ref.nbytes,
        "byte_ratio": byte_ratio,
        "bytes_pass": bytes_pass,
        "quantize_meta": qmeta,
        "monarch_probe": monarch_meta,
        "spectrum_decision": spec.get("decision"),
        "hard_constraints": "docs/hard-constraints.md",
        "note": "Fail must not be patched by lowering bits on untouched tensors.",
    }
    _dump(out_dir / "metrics.json", result)
    _dump(art / "phase3_status.json", result)
    return result


def phase4_fuse_time(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    p3 = json.loads((art / "phase3_status.json").read_text())
    layer = int(p3["layer"])
    fac = np.load(art / "down_proj_prototype" / f"layer{layer}_factors.npz")
    from hemifactor.factorize.hemifactor import Hemifactor

    hf = Hemifactor(A=fac["A"], B=fac["B"], k=int(p3["k"]), role="down", layer=layer)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(hf.B.shape[1], dtype=np.float64)
    bench = bench_fused(hf, x, repeats=100)
    hw = cfg["hardware"]
    traffic = bytes_per_token(
        weight_or_factor_bytes=float(p3["factor_nbytes_est"]),
        kv_bytes=float(hw["swa_kv_mib_estimate"]) * 1024 * 1024 / max(int(cfg["model"]["n_layers"]), 1),
    )
    # role-local traffic vs role baseline
    baseline_role = float(p3["baseline_nbytes"])
    roof = compare_buses(
        bytes_per_token=float(p3["factor_nbytes_est"]),
        gddr_gbs=float(hw["gddr_gbps"]),
        pcie_gbs=float(hw["pcie_read_gbps_estimate"]),
    )
    roof_base = compare_buses(
        bytes_per_token=baseline_role,
        gddr_gbs=float(hw["gddr_gbps"]),
        pcie_gbs=float(hw["pcie_read_gbps_estimate"]),
    )
    out = {
        "phase": 4,
        "status": "ok",
        "bench": bench,
        "role_traffic": traffic,
        "roofline_factor": roof,
        "roofline_baseline_role": roof_base,
        "speedup_role_roofline": roof["tok_s_gddr"] / max(roof_base["tok_s_gddr"], 1e-9),
        "caveat": "Microbenchmark is CPU NumPy fused path; on-device CUDA timing still required for tok/s claims.",
        "power_limits_w": [hw["power_limit_w_default"], hw["power_limit_w_max"]],
    }
    _dump(art / "phase4_status.json", out)
    return out


def phase5_assemble(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    art = artifacts_dir(cfg)
    freeze = json.loads((art / "model_freeze.json").read_text())
    p2 = json.loads((art / "phase2_status.json").read_text())
    p3 = json.loads((art / "phase3_status.json").read_text())
    p4 = json.loads((art / "phase4_status.json").read_text())
    hw = cfg["hardware"]
    inv = freeze["inventory_roles"]
    down_bytes = float(inv["down"]["nbytes"])
    # Extrapolate prototype byte ratio to all downs only; other roles untouched
    compressed_down = down_bytes * float(p3["byte_ratio"])
    untouched = freeze["tensor_nbytes_sum"] - down_bytes
    total_weight = compressed_down + untouched
    kv = float(hw["swa_kv_mib_estimate"])
    usable = float(hw["vram_usable_mib_estimate"]) - kv
    working_mib = total_weight / (1024 * 1024)
    traffic = bytes_per_token(total_weight, kv_bytes=kv * 1024 * 1024)
    baseline = bytes_per_token(float(freeze["tensor_nbytes_sum"]), kv_bytes=kv * 1024 * 1024)
    r = ratio_vs_baseline(traffic["bytes_per_token"], baseline["bytes_per_token"])
    success = {
        "distortion_bar": bool(p3.get("quality_pass")),
        "e2e_nll_bar": False,  # not measured without full forward harness
        "traffic_bar": r <= float(cfg["project"]["traffic_ratio_acceptable"]),
        "residency_bar": residency_ok(working_mib + kv, float(hw["vram_usable_mib_estimate"])),
        "speed_bar": False,  # requires on-device measurement
    }
    # Honest aggregate: cannot claim success predicate without e2e NLL + device timing
    predicate_pass = all(success.values())
    out = {
        "phase": 5,
        "status": "complete",
        "spectrum_aggregate": p2.get("aggregate"),
        "prototype": {"status": p3.get("status"), "byte_ratio": p3.get("byte_ratio"), "quality_pass": p3.get("quality_pass")},
        "extrapolated_down_only_compression": {
            "working_set_mib": working_mib,
            "with_swa_kv_mib": working_mib + kv,
            "traffic_ratio_vs_full_file": r,
            "residency_ok_estimate": success["residency_bar"],
        },
        "success_predicate": success,
        "success_predicate_pass": predicate_pass,
        "phase4_caveat": p4.get("caveat"),
        "hard_constraints": "docs/hard-constraints.md",
        "conclusion": (
            "Do not claim ~2× tok/s or full-model success: e2e NLL and GDDR-resident "
            "device timing are not cleared. Record spectrum/prototype outcomes as-is."
        ),
    }
    _dump(art / "phase5_status.json", out)
    results_md = Path(cfg["_root"]) / "docs" / "results.md"
    results_md.write_text(
        "\n".join(
            [
                "# Results",
                "",
                f"Spectrum aggregate: **{p2.get('aggregate')}**",
                f"Down-proj prototype: **{p3.get('status')}** (byte_ratio={p3.get('byte_ratio'):.4f}, quality_pass={p3.get('quality_pass')})",
                f"Success predicate pass: **{predicate_pass}**",
                "",
                "See `artifacts/phase*_status.json` and `artifacts/rank_license.md`.",
                "",
                out["conclusion"],
                "",
            ]
        )
    )
    return out


def run_all(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    results = {
        "phase0": phase0_freeze(cfg),
        "phase1": phase1_calibrate(cfg),
        "phase2": phase2_spectrum(cfg),
        "phase3": phase3_down_prototype(cfg),
        "phase4": phase4_fuse_time(cfg),
        "phase5": phase5_assemble(cfg),
    }
    return results
