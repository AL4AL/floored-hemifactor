"""Unit tests for spectrum / factor / constraint math (no GGUF required)."""

from __future__ import annotations

import numpy as np
import pytest

from hemifactor.calibrate.distortion import activation_mse
from hemifactor.calibrate.welford import WelfordCov
from hemifactor.constraints import lint_success_text
from hemifactor.eval.metrics import within_control
from hemifactor.eval.roofline import tok_s_from_bytes
from hemifactor.factorize.hemifactor import from_truncated_svd
from hemifactor.factorize.quantize_factors import quantize_hemifactor
from hemifactor.factorize.whitened import hemifactor_whitened
from hemifactor.spectrum.svd_license import discarded_energy, license_rank
from hemifactor.spectrum.whiten import whitened_singular_values


def test_welford_participation_identity_like():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 32))
    w = WelfordCov(32)
    w.update_batch(x)
    pr = w.participation_ratio()
    assert 10 < pr < 40


def test_discarded_energy_and_license():
    s = np.array([5.0, 4.0, 0.1, 0.05])
    assert discarded_energy(s, 2) == pytest.approx(0.1**2 + 0.05**2)
    lic = license_rank(s, d_floor=0.02, candidate_ks=[1, 2, 3])
    assert lic["chosen_k"] == 2


def test_hemifactor_recon_low_rank():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((40, 5))
    B = rng.standard_normal((5, 30))
    W = A @ B
    hf = from_truncated_svd(W, k=5, role="down")
    err = np.linalg.norm(W - hf.A @ hf.B)
    assert err < 1e-6


def test_whitened_diag_matches_scaled_svd():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((20, 40))
    diag = rng.random(40) + 0.1
    s = whitened_singular_values(W, diag_var=diag)
    M = W * np.sqrt(diag)[None, :]
    s2 = np.linalg.svd(M, compute_uv=False)
    np.testing.assert_allclose(s, s2, rtol=1e-6)


def test_whitened_hemifactor_beats_naive_on_colored_mu():
    rng = np.random.default_rng(3)
    n, m, k = 64, 32, 8
    # colored inputs
    diag = np.linspace(1.0, 0.01, n)
    x = rng.standard_normal((400, n)) * np.sqrt(diag)
    W = rng.standard_normal((m, n))
    hf = hemifactor_whitened(W, k=k, role="down", diag_var=diag)
    naive = from_truncated_svd(W, k=k, role="down")
    mse_w = activation_mse(W, hf.A @ hf.B, x)
    mse_n = activation_mse(W, naive.A @ naive.B, x)
    assert mse_w <= mse_n * 1.05


def test_quantize_factors_bits():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((16, 24))
    hf = from_truncated_svd(W, k=4, role="down")
    hq, meta = quantize_hemifactor(hf, bits=4)
    assert meta["nbytes_est"] > 0
    assert hq.A.shape == hf.A.shape


def test_lint_forbids_lossless_claim():
    hits = lint_success_text("we losslessly fit exact GGUF into 6GB")
    assert hits


def test_within_control_and_roofline():
    assert within_control(0.1, 0.2)
    assert not within_control(0.3, 0.2)
    assert tok_s_from_bytes(168e9 / 42, 168.0) == pytest.approx(42.0, rel=1e-6)
