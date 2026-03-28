from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


def sample_disk(n_points: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_points))
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta)))


def sample_bimodal(n_points: int, offset: float, spread: float, rng: np.random.Generator) -> np.ndarray:
    half = n_points // 2
    cloud1 = rng.normal(loc=(-offset, 0.0), scale=spread, size=(half, 2))
    cloud2 = rng.normal(loc=(offset, 0.0), scale=spread, size=(n_points - half, 2))
    return np.vstack((cloud1, cloud2))


def rk4_step(pos: np.ndarray, t: float, dt: float, velocity_fn) -> np.ndarray:
    k1 = velocity_fn(pos, t)
    k2 = velocity_fn(pos + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = velocity_fn(pos + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = velocity_fn(pos + dt * k3, t + dt)
    return pos + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(initial_pos: np.ndarray, times: np.ndarray, velocity_fn) -> np.ndarray:
    frames = np.empty((len(times), len(initial_pos), 2))
    frames[0] = initial_pos
    current = initial_pos.copy()
    for i in range(1, len(times)):
        current = rk4_step(current, times[i - 1], times[i] - times[i - 1], velocity_fn)
        frames[i] = current
    return frames


def sinusoidal_folding_velocity(pos: np.ndarray, t: float) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    return np.column_stack((0.16 * np.sin(0.6 * t) + 0.10 * y, 0.55 * np.sin(1.3 * x + 0.8 * t) - 0.04 * x))


def four_way_split_velocity(pos: np.ndarray, t: float) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    return np.column_stack(
        (
            0.35 * np.tanh(2.2 * x) + 0.18 * np.sin(0.9 * y + 0.4 * t),
            0.35 * np.tanh(2.2 * y) + 0.18 * np.cos(0.9 * x - 0.5 * t),
        )
    )


def annular_filament_velocity(pos: np.ndarray, t: float) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    r = np.sqrt(x * x + y * y) + 1e-12
    theta = np.arctan2(y, x)
    swirl = np.column_stack((-0.55 * y / (0.35 + r * r), 0.55 * x / (0.35 + r * r)))
    radial = np.column_stack((0.22 * (r - 1.8) * x / r, 0.22 * (r - 1.8) * y / r))
    waviness = np.column_stack((0.08 * np.sin(3.0 * theta + 0.5 * t), 0.08 * np.cos(3.0 * theta - 0.4 * t)))
    return swirl + radial + waviness


def saddle_twist_velocity(pos: np.ndarray, t: float) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    return np.column_stack(
        (
            0.38 * x - 0.30 * y + 0.14 * np.sin(1.1 * x * y + 0.3 * t),
            -0.34 * y + 0.26 * x + 0.14 * np.cos(0.9 * x * y - 0.4 * t),
        )
    )


SCENARIOS = [
    {
        "slug": "dobramento_senoidal",
        "title": "Dobramento Senoidal",
        "builder": lambda rng: sample_disk(220, 1.1, rng),
        "velocity": sinusoidal_folding_velocity,
    },
    {
        "slug": "cisao_em_quatro",
        "title": "Cisao em Quatro",
        "builder": lambda rng: sample_disk(240, 1.0, rng),
        "velocity": four_way_split_velocity,
    },
    {
        "slug": "anel_filamentado",
        "title": "Anel Filamentado",
        "builder": lambda rng: sample_disk(220, 0.9, rng) + np.array([0.8, 0.0]),
        "velocity": annular_filament_velocity,
    },
    {
        "slug": "sela_torcida",
        "title": "Sela Torcida",
        "builder": lambda rng: sample_disk(220, 1.0, rng),
        "velocity": saddle_twist_velocity,
    },
    {
        "slug": "multimodal_leve",
        "title": "Multimodal Leve",
        "builder": lambda rng: sample_bimodal(220, 1.5, 0.45, rng),
        "velocity": sinusoidal_folding_velocity,
    },
]


def matrix_sqrt_spd(cov: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def affine_moments(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.mean(pos, axis=0)
    rel = pos - q
    cov = (rel.T @ rel) / len(pos)
    a = matrix_sqrt_spd(cov)
    return q, a


def quadratic_features(z: np.ndarray) -> np.ndarray:
    return np.column_stack((z[:, 0] ** 2 - 1.0, z[:, 0] * z[:, 1], z[:, 1] ** 2 - 1.0))


def third_moments(z: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.mean(z[:, 0] ** 3),
            np.mean((z[:, 0] ** 2) * z[:, 1]),
            np.mean(z[:, 0] * (z[:, 1] ** 2)),
            np.mean(z[:, 1] ** 3),
        ],
        dtype=float,
    )


def mean_cov_errors(pos: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mean_err = float(np.linalg.norm(np.mean(pos, axis=0) - np.mean(pred, axis=0)))
    pos_cov = np.cov(pos.T, bias=True)
    pred_cov = np.cov(pred.T, bias=True)
    cov_err = float(np.linalg.norm(pos_cov - pred_cov))
    return mean_err, cov_err


def fit_moment_gauge(z0: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    base_m = third_moments(z0)
    target_m = third_moments(target_norm)
    rhs = target_m - base_m
    eps = 1e-3
    h = quadratic_features(z0)
    jac = np.zeros((4, 6), dtype=float)
    for out_dim in range(2):
        for feat_idx in range(3):
            col = 3 * out_dim + feat_idx
            pert = np.zeros_like(z0)
            pert[:, out_dim] = h[:, feat_idx]
            deriv = (third_moments(z0 + eps * pert) - base_m) / eps
            jac[:, col] = deriv
    coeffs, _, _, _ = np.linalg.lstsq(jac, rhs, rcond=None)
    return coeffs.reshape(2, 3)


def predict_moment_like(z0: np.ndarray, q: np.ndarray, a: np.ndarray, c: np.ndarray) -> np.ndarray:
    h = quadratic_features(z0)
    y = z0 + h @ c.T
    return q + y @ a.T


def fit_hybrid_gauge(z0: np.ndarray, target_norm: np.ndarray) -> np.ndarray:
    h = quadratic_features(z0)
    delta = target_norm - z0
    coeffs, _, _, _ = np.linalg.lstsq(h, delta, rcond=None)
    return coeffs.T


def fit_minimization_gauge(z0: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(z0)), z0[:, 0], z0[:, 1], z0[:, 0] ** 2, z0[:, 0] * z0[:, 1], z0[:, 1] ** 2))
    coeffs_x, _, _, _ = np.linalg.lstsq(design, pos[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(design, pos[:, 1], rcond=None)
    return coeffs_x, coeffs_y


def predict_minimization_gauge(z0: np.ndarray, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(z0)), z0[:, 0], z0[:, 1], z0[:, 0] ** 2, z0[:, 0] * z0[:, 1], z0[:, 1] ** 2))
    return np.column_stack((design @ coeffs_x, design @ coeffs_y))


def jacobian_negativity_fraction_moment_like(z0: np.ndarray, c: np.ndarray) -> float:
    z1 = z0[:, 0]
    z2 = z0[:, 1]
    g11 = 1.0 + 2.0 * c[0, 0] * z1 + c[0, 1] * z2
    g12 = c[0, 1] * z1 + 2.0 * c[0, 2] * z2
    g21 = 2.0 * c[1, 0] * z1 + c[1, 1] * z2
    g22 = 1.0 + c[1, 1] * z1 + 2.0 * c[1, 2] * z2
    det = g11 * g22 - g12 * g21
    return float(np.mean(det <= 0.0))


def jacobian_negativity_fraction_minimization(z0: np.ndarray, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> float:
    z1 = z0[:, 0]
    z2 = z0[:, 1]
    dx_dz1 = coeffs_x[1] + 2.0 * coeffs_x[3] * z1 + coeffs_x[4] * z2
    dx_dz2 = coeffs_x[2] + coeffs_x[4] * z1 + 2.0 * coeffs_x[5] * z2
    dy_dz1 = coeffs_y[1] + 2.0 * coeffs_y[3] * z1 + coeffs_y[4] * z2
    dy_dz2 = coeffs_y[2] + coeffs_y[4] * z1 + 2.0 * coeffs_y[5] * z2
    det = dx_dz1 * dy_dz2 - dx_dz2 * dy_dz1
    return float(np.mean(det <= 0.0))


@dataclass
class GaugeMetrics:
    scenario: str
    seed: int
    gauge: str
    reconstruction_error: float
    mean_error: float
    covariance_error: float
    third_moment_error: float
    negative_jacobian_fraction: float


def evaluate_scenario(initial: np.ndarray, final: np.ndarray, scenario: str, seed: int) -> list[GaugeMetrics]:
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T

    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    target_third = third_moments(target_norm)

    metrics: list[GaugeMetrics] = []

    c_mom = fit_moment_gauge(z0, target_norm)
    pred_mom = predict_moment_like(z0, qf, af, c_mom)
    mean_err, cov_err = mean_cov_errors(final, pred_mom)
    third_err = float(np.linalg.norm(third_moments(np.linalg.solve(af, (pred_mom - qf).T).T) - target_third))
    metrics.append(
        GaugeMetrics(
            scenario=scenario,
            seed=seed,
            gauge="momentos",
            reconstruction_error=float(np.sqrt(np.mean(np.sum((final - pred_mom) ** 2, axis=1)))),
            mean_error=mean_err,
            covariance_error=cov_err,
            third_moment_error=third_err,
            negative_jacobian_fraction=jacobian_negativity_fraction_moment_like(z0, c_mom),
        )
    )

    c_hybrid = fit_hybrid_gauge(z0, target_norm)
    pred_hybrid = predict_moment_like(z0, qf, af, c_hybrid)
    mean_err, cov_err = mean_cov_errors(final, pred_hybrid)
    third_err = float(np.linalg.norm(third_moments(np.linalg.solve(af, (pred_hybrid - qf).T).T) - target_third))
    metrics.append(
        GaugeMetrics(
            scenario=scenario,
            seed=seed,
            gauge="hibrida",
            reconstruction_error=float(np.sqrt(np.mean(np.sum((final - pred_hybrid) ** 2, axis=1)))),
            mean_error=mean_err,
            covariance_error=cov_err,
            third_moment_error=third_err,
            negative_jacobian_fraction=jacobian_negativity_fraction_moment_like(z0, c_hybrid),
        )
    )

    coeffs_x, coeffs_y = fit_minimization_gauge(z0, final)
    pred_min = predict_minimization_gauge(z0, coeffs_x, coeffs_y)
    mean_err, cov_err = mean_cov_errors(final, pred_min)
    pred_q, pred_a = affine_moments(pred_min)
    pred_norm = np.linalg.solve(pred_a, (pred_min - pred_q).T).T
    third_err = float(np.linalg.norm(third_moments(pred_norm) - target_third))
    metrics.append(
        GaugeMetrics(
            scenario=scenario,
            seed=seed,
            gauge="minimizacao",
            reconstruction_error=float(np.sqrt(np.mean(np.sum((final - pred_min) ** 2, axis=1)))),
            mean_error=mean_err,
            covariance_error=cov_err,
            third_moment_error=third_err,
            negative_jacobian_fraction=jacobian_negativity_fraction_minimization(z0, coeffs_x, coeffs_y),
        )
    )
    return metrics


def run_suite(n_seeds: int = 16) -> list[GaugeMetrics]:
    rows: list[GaugeMetrics] = []
    times = np.linspace(0.0, 10.0, 180)
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario["builder"](rng)
            frames = simulate(initial, times, scenario["velocity"])
            final = frames[-1]
            rows.extend(evaluate_scenario(initial, final, scenario["slug"], seed))
    return rows


def write_outputs(base_dir: Path, rows: list[GaugeMetrics]) -> None:
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    (out_dir / "gauge_comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    grouped: dict[str, dict[str, list[GaugeMetrics]]] = {}
    for row in rows:
        grouped.setdefault(row.scenario, {}).setdefault(row.gauge, []).append(row)

    global_by_gauge: dict[str, list[GaugeMetrics]] = {}
    for row in rows:
        global_by_gauge.setdefault(row.gauge, []).append(row)

    lines = ["# Comparacao de Gauges para V3", ""]
    lines.append("## Resultado Global")
    lines.append("")
    for gauge, items in global_by_gauge.items():
        lines.append(
            f"- `{gauge}`: erro={np.mean([x.reconstruction_error for x in items]):.3e}, "
            f"media={np.mean([x.mean_error for x in items]):.3e}, "
            f"cov={np.mean([x.covariance_error for x in items]):.3e}, "
            f"momento3={np.mean([x.third_moment_error for x in items]):.3e}, "
            f"jac_neg={np.mean([x.negative_jacobian_fraction for x in items]):.2%}"
        )

    lines.extend(["", "## Resultado por Cenario", ""])
    for scenario, by_gauge in grouped.items():
        lines.append(f"### `{scenario}`")
        lines.append("")
        summary = []
        for gauge, items in by_gauge.items():
            summary.append(
                (
                    gauge,
                    np.mean([x.reconstruction_error for x in items]),
                    np.mean([x.mean_error for x in items]),
                    np.mean([x.covariance_error for x in items]),
                    np.mean([x.third_moment_error for x in items]),
                    np.mean([x.negative_jacobian_fraction for x in items]),
                )
            )
        summary.sort(key=lambda item: item[1])
        for gauge, rec, mean_err, cov_err, third_err, jac_neg in summary:
            lines.append(
                f"- `{gauge}`: erro={rec:.3e}, media={mean_err:.3e}, cov={cov_err:.3e}, "
                f"momento3={third_err:.3e}, jac_neg={jac_neg:.2%}"
            )
        lines.append("")

    lines.extend(
        [
            "## Leitura",
            "",
            "- `momentos` usa apenas estrutura estatistica para fixar a parte nao linear; tende a preservar melhor a interpretacao canonica, mas pode perder em erro ponto a ponto.",
            "- `minimizacao` busca o melhor ajuste geometrico na classe escolhida; tende a reduzir o erro de reconstrucao, mas pode sacrificar canonicidade e estabilidade do mapa.",
            "- `hibrida` fixa a base afim por momentos e ajusta apenas a parte nao linear por minimizacao; a expectativa e que ela funcione como compromisso pratico.",
        ]
    )
    (out_dir / "gauge_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_suite()
    write_outputs(base_dir, rows)
    print("Comparacao de gauges V3 concluida.")
    print(f"Saidas em {base_dir / 'results'}")


if __name__ == "__main__":
    main()
