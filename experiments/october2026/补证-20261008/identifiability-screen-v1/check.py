"""Exact Gaussian screening; classical identities, no image efficacy claim."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parent
started = time.monotonic()
protocol_bytes = (ROOT / "protocol.json").read_bytes()
protocol = json.loads(protocol_bytes)


def posterior(C, A, nu):
    S = A @ C @ A.T + nu * np.eye(len(A))
    H = np.linalg.solve(S, A @ C).T
    return S, H


def risk(C, A, nu, H):
    E = np.eye(2) - H @ A
    return float(np.trace(E @ C @ E.T + nu * H @ H.T))


def kl(S1, S2):
    # Zero-mean Gaussian KL, one draw of the entire observation vector.
    return float(0.5 * (np.trace(np.linalg.solve(S2, S1)) - len(S1)
                       + np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1]))


single = []
interventions = []
checks = []
nodes, weights = np.polynomial.hermite.hermgauss(96)
weights /= np.sqrt(np.pi)
for p in protocol["power_p"]:
    C1, C2 = np.diag([p, 1.0]), np.diag([1.0, p])
    for nu in protocol["noise_variance_nu"]:
        A = np.array([[1., 1.]])
        S1, H1 = posterior(C1, A, nu)
        S2, H2 = posterior(C2, A, nu)
        s = p + 1 + nu
        J = 2 * (p - 1)**2 / s
        matrix_J = float(np.trace((H1-H2) @ S1 @ (H1-H2).T))
        oracle1, oracle2 = risk(C1, A, nu, H1), risk(C2, A, nu, H2)
        wrong = risk(C1, A, nu, H2) - oracle1
        midpoint = (H1 + H2) / 2
        avg_mid_excess = ((risk(C1, A, nu, midpoint) - oracle1)
                          + (risk(C2, A, nu, midpoint) - oracle2)) / 2
        y = np.sqrt(2*s) * nodes
        m1, m2 = y[:, None]*H1.T, y[:, None]*H2.T
        selectors = {"constant_arm1": np.ones(len(y), dtype=bool),
                     "sign_y": y >= 0,
                     "absolute_y_gt_std": np.abs(y) > np.sqrt(s)}
        quadrature = {}
        for name, arm1 in selectors.items():
            estimate = np.where(arm1[:, None], m1, m2)
            # Orthogonality to each true Gaussian conditional mean; equal
            # observational laws make posterior model weights exactly 1/2.
            excess = (np.sum((estimate-m1)**2, axis=1)
                      + np.sum((estimate-m2)**2, axis=1)) / 2
            quadrature[name] = float(weights @ excess)
        errors = [float(np.max(np.abs(S1-S2))), abs(matrix_J-J), abs(wrong-J),
                  abs(avg_mid_excess-J/4),
                  *[abs(value-J/2) for value in quadrature.values()]]
        checks.extend(errors)
        single.append({"p": p, "nu": nu, "observation_covariance": S1.tolist(),
                       "KL_1_to_2": kl(S1, S2), "oracle_mse": oracle1,
                       "wrong_arm_excess_J": J,
                       "equal_model_minimum_average_excess_any_estimator": J/4,
                       "equal_model_average_excess_any_two_arm_selector": J/2,
                       "selector_quadrature": quadrature, "max_error": max(errors)})
        for delta in protocol["second_measurement_delta"]:
            A2 = np.array([[1., 1.], [1., 1.-delta]])
            T1, G1 = posterior(C1, A2, nu)
            T2, G2 = posterior(C2, A2, nu)
            excess1 = risk(C1, A2, nu, G2) - risk(C1, A2, nu, G1)
            identity1 = float(np.trace((G2-G1) @ T1 @ (G2-G1).T))
            checks.append(abs(excess1-identity1))
            interventions.append({"p": p, "nu": nu, "delta": delta,
                                  "A_rank": int(np.linalg.matrix_rank(A2)),
                                  "covariance_distance_fro": float(np.linalg.norm(T1-T2)),
                                  "KL_1_to_2": kl(T1, T2), "KL_2_to_1": kl(T2, T1),
                                  "wrong_arm_excess_model1": excess1,
                                  "oracle_mse_model1": risk(C1, A2, nu, G1)})

elapsed = time.monotonic()-started
result = {"protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
          "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "identity_tolerance": 1e-10, "max_absolute_identity_error": max(checks),
          "all_identities_pass": max(checks) < 1e-10,
          "cpu_wall_seconds": elapsed, "within_budget": elapsed < protocol["budget"]["cpu_wall_seconds"],
          "single_rows": single, "intervention_rows": interventions,
          "interpretation": "Classical nonidentifiability reproduced in a two-component Gaussian toy. Equal observable covariance does not imply equal posterior means. Acquisition can make covariance models distinguishable, not necessarily reliably selectable from finite observations. No new-theorem, INR-mechanism or real-image efficacy claim."}
(ROOT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({key: result[key] for key in ["all_identities_pass", "max_absolute_identity_error", "cpu_wall_seconds", "within_budget"]}))
assert result["all_identities_pass"] and result["within_budget"]
