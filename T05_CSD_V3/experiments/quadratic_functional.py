from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np

from gauge_comparison import (
    SCENARIOS,
    affine_moments,
    jacobian_negativity_fraction_moment_like,
    mean_cov_errors,
    predict_moment_like,
    quadratic_features,
    simulate,
    third_moments,
)


@dataclass
class QuadraticFunctionalWeights:
    reconstruction: float = 1.0
    moment3: float = 0.15
    affine_orthogonality: float = 0.25
    jacobian: float = 2.0
    regularization: float = 1e-3
    jacobian_buffer: float = 0.10
    jacobian_hardness: float = 6.0


@dataclass
class QuadraticFunctionalResult:
    scenario: str
    seed: int
    gauge: str
    total: float
    reconstruction_term: float
    moment3_term: float
    affine_term: float
    jacobian_term: float
    regularization_term: float
    reconstruction_error: float
    mean_error: float
    covariance_error: float
    third_moment_error: float
    negative_jacobian_fraction: float
    tensor_norm: float


@dataclass
class FamilyBasis:
    name: str
    values: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray


def build_family_basis(z: np.ndarray, family: str = "quadratic") -> FamilyBasis:
    z1 = z[:, 0]
    z2 = z[:, 1]
    if family == "quadratic":
        values = np.column_stack((z1 * z1 - 1.0, z1 * z2, z2 * z2 - 1.0))
        grad_x = np.column_stack((2.0 * z1, z2, np.zeros_like(z1)))
        grad_y = np.column_stack((np.zeros_like(z1), z1, 2.0 * z2))
        return FamilyBasis(name=family, values=values, grad_x=grad_x, grad_y=grad_y)
    if family == "cubic":
        values = np.column_stack(
            (
                z1 * z1 - 1.0,
                z1 * z2,
                z2 * z2 - 1.0,
                z1 * z1 * z1 - 3.0 * z1,
                z1 * z1 * z2 - z2,
                z1 * z2 * z2 - z1,
                z2 * z2 * z2 - 3.0 * z2,
            )
        )
        grad_x = np.column_stack(
            (
                2.0 * z1,
                z2,
                np.zeros_like(z1),
                3.0 * z1 * z1 - 3.0,
                2.0 * z1 * z2,
                z2 * z2 - 1.0,
                np.zeros_like(z1),
            )
        )
        grad_y = np.column_stack(
            (
                np.zeros_like(z1),
                z1,
                2.0 * z2,
                np.zeros_like(z1),
                z1 * z1 - 1.0,
                2.0 * z1 * z2,
                3.0 * z2 * z2 - 3.0,
            )
        )
        return FamilyBasis(name=family, values=values, grad_x=grad_x, grad_y=grad_y)
    raise ValueError(f"Unsupported family: {family}")


def affine_orthogonality_mismatch(z: np.ndarray, c: np.ndarray) -> float:
    if c.shape[1] == 3:
        h = quadratic_features(z)
    else:
        h = build_family_basis(z, "cubic").values
    n = h @ c.T
    t00 = np.mean(z[:, 0] * n[:, 0])
    t01 = np.mean(z[:, 0] * n[:, 1])
    t10 = np.mean(z[:, 1] * n[:, 0])
    t11 = np.mean(z[:, 1] * n[:, 1])
    return float(np.sqrt(t00 * t00 + t01 * t01 + t10 * t10 + t11 * t11))


def jacobian_determinants_from_basis(basis: FamilyBasis, c: np.ndarray) -> np.ndarray:
    g11 = 1.0 + basis.grad_x @ c[0]
    g12 = basis.grad_y @ c[0]
    g21 = basis.grad_x @ c[1]
    g22 = 1.0 + basis.grad_y @ c[1]
    return g11 * g22 - g12 * g21


def jacobian_determinants(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    family = "quadratic" if c.shape[1] == 3 else "cubic"
    return jacobian_determinants_from_basis(build_family_basis(z, family), c)


def jacobian_penalty_from_basis(basis: FamilyBasis, c: np.ndarray, margin: float = 0.15) -> float:
    det = jacobian_determinants_from_basis(basis, c)
    bad = np.maximum(margin - det, 0.0)
    return float(np.mean(bad * bad))


def jacobian_penalty(z: np.ndarray, c: np.ndarray, margin: float = 0.15) -> float:
    family = "quadratic" if c.shape[1] == 3 else "cubic"
    return jacobian_penalty_from_basis(build_family_basis(z, family), c, margin)


def jacobian_linearization_from_basis(basis: FamilyBasis, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g11 = 1.0 + basis.grad_x @ c[0]
    g12 = basis.grad_y @ c[0]
    g21 = basis.grad_x @ c[1]
    g22 = 1.0 + basis.grad_y @ c[1]
    base = g11 * g22 - g12 * g21
    n_features = basis.values.shape[1]
    jac = np.zeros((len(base), 2 * n_features), dtype=float)
    jac[:, :n_features] = basis.grad_x * g22[:, None] - basis.grad_y * g21[:, None]
    jac[:, n_features:] = basis.grad_y * g11[:, None] - basis.grad_x * g12[:, None]
    return base, jac


def jacobian_linearization(z: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    family = "quadratic" if c.shape[1] == 3 else "cubic"
    return jacobian_linearization_from_basis(build_family_basis(z, family), c)


def evaluate_quadratic_functional(
    z0: np.ndarray,
    qf: np.ndarray,
    af: np.ndarray,
    final: np.ndarray,
    c: np.ndarray,
    weights: QuadraticFunctionalWeights,
    scenario: str,
    seed: int,
    gauge: str,
) -> QuadraticFunctionalResult:
    pred = predict_moment_like(z0, qf, af, c)
    rec_err = float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))
    mean_err, cov_err = mean_cov_errors(final, pred)

    target_norm = np.linalg.solve(af, (final - qf).T).T
    target_third = third_moments(target_norm)
    pred_norm = np.linalg.solve(af, (pred - qf).T).T
    third_err = float(np.linalg.norm(third_moments(pred_norm) - target_third))

    affine_err = affine_orthogonality_mismatch(z0, c)
    jac_pen = jacobian_penalty(z0, c)
    reg = float(np.sum(c * c))
    total = (
        weights.reconstruction * rec_err * rec_err
        + weights.moment3 * third_err * third_err
        + weights.affine_orthogonality * affine_err * affine_err
        + weights.jacobian * jac_pen
        + weights.regularization * reg
    )

    return QuadraticFunctionalResult(
        scenario=scenario,
        seed=seed,
        gauge=gauge,
        total=float(total),
        reconstruction_term=float(weights.reconstruction * rec_err * rec_err),
        moment3_term=float(weights.moment3 * third_err * third_err),
        affine_term=float(weights.affine_orthogonality * affine_err * affine_err),
        jacobian_term=float(weights.jacobian * jac_pen),
        regularization_term=float(weights.regularization * reg),
        reconstruction_error=rec_err,
        mean_error=mean_err,
        covariance_error=cov_err,
        third_moment_error=third_err,
        negative_jacobian_fraction=jacobian_negativity_fraction_moment_like(z0, c),
        tensor_norm=float(np.linalg.norm(c)),
    )


def base_linear_system_from_basis(
    z0: np.ndarray,
    target_norm: np.ndarray,
    basis: FamilyBasis,
    weights: QuadraticFunctionalWeights,
) -> tuple[np.ndarray, np.ndarray]:
    h = basis.values
    delta = target_norm - z0
    n_features = h.shape[1]

    design = np.zeros((2 * len(z0), 2 * n_features), dtype=float)
    design[: len(z0), :n_features] = h
    design[len(z0) :, n_features:] = h
    rhs = np.concatenate((delta[:, 0], delta[:, 1]))
    gram = weights.reconstruction * (design.T @ design)
    vec = weights.reconstruction * (design.T @ rhs)

    moments = np.column_stack((np.mean(z0[:, 0:1] * h, axis=0), np.mean(z0[:, 1:2] * h, axis=0))).T
    cmat_rows: list[np.ndarray] = []
    for out_dim in range(2):
        for l in range(2):
            row = np.zeros(2 * n_features)
            row[n_features * out_dim : n_features * out_dim + n_features] = moments[l]
            cmat_rows.append(row)
    cmat = np.vstack(cmat_rows)
    gram += weights.affine_orthogonality * (cmat.T @ cmat)

    amat = np.zeros((2, 2 * n_features), dtype=float)
    amat[0, 0] = 1.0
    amat[0, min(2, n_features - 1)] = 1.0
    amat[1, n_features] = 1.0
    amat[1, n_features + min(2, n_features - 1)] = 1.0
    gram += 10.0 * weights.affine_orthogonality * (amat.T @ amat)

    eps = 1e-4
    base_m = third_moments(z0)
    target_m = third_moments(target_norm)
    jac = np.zeros((4, 2 * n_features), dtype=float)
    for out_dim in range(2):
        for feat_idx in range(n_features):
            col = n_features * out_dim + feat_idx
            pert = np.zeros_like(z0)
            pert[:, out_dim] = h[:, feat_idx]
            jac[:, col] = (third_moments(z0 + eps * pert) - base_m) / eps
    gram += weights.moment3 * (jac.T @ jac)
    vec += weights.moment3 * (jac.T @ (target_m - base_m))

    gram += weights.regularization * np.eye(2 * n_features)
    return gram, vec


def base_linear_system(z0: np.ndarray, target_norm: np.ndarray, weights: QuadraticFunctionalWeights) -> tuple[np.ndarray, np.ndarray]:
    return base_linear_system_from_basis(z0, target_norm, build_family_basis(z0, "quadratic"), weights)


def fit_penalized_family(
    z0: np.ndarray,
    target_norm: np.ndarray,
    weights: QuadraticFunctionalWeights,
    family: str = "quadratic",
    jacobian_margin: float = 0.15,
    max_iter: int = 8,
) -> np.ndarray:
    basis = build_family_basis(z0, family)
    base_gram, base_vec = base_linear_system_from_basis(z0, target_norm, basis, weights)
    n_features = basis.values.shape[1]
    cvec = np.linalg.solve(base_gram + 1e-8 * np.eye(2 * n_features), base_vec)

    for _ in range(max_iter):
        cmat = cvec.reshape(2, n_features)
        det0, det_jac = jacobian_linearization_from_basis(basis, cmat)
        active = det0 < (jacobian_margin + weights.jacobian_buffer)
        if not np.any(active) or weights.jacobian <= 0.0:
            break

        det_active = det0[active]
        j_active = det_jac[active]
        deficit = np.maximum(jacobian_margin - det_active, 0.0)
        proximity = np.maximum(jacobian_margin + weights.jacobian_buffer - det_active, 0.0)
        point_weights = 1.0 + weights.jacobian_hardness * (proximity / (jacobian_margin + weights.jacobian_buffer + 1e-12)) ** 2
        point_weights += 4.0 * weights.jacobian_hardness * (deficit / (jacobian_margin + 1e-12)) ** 2

        target = np.full(np.sum(active), jacobian_margin) + 0.5 * deficit
        residual_target = target - det_active + j_active @ cvec

        weight_sum = max(float(np.sum(point_weights)), 1.0)
        weighted_j = point_weights[:, None] * j_active
        gram = base_gram + weights.jacobian * (j_active.T @ weighted_j) / weight_sum
        vec = base_vec + weights.jacobian * (j_active.T @ (point_weights * residual_target)) / weight_sum
        new_cvec = np.linalg.solve(gram + 1e-8 * np.eye(2 * n_features), vec)

        if np.linalg.norm(new_cvec - cvec) <= 1e-7 * (1.0 + np.linalg.norm(cvec)):
            cvec = new_cvec
            break
        cvec = new_cvec

    return cvec.reshape(2, n_features)


def fit_penalized_tensor(
    z0: np.ndarray,
    target_norm: np.ndarray,
    weights: QuadraticFunctionalWeights,
    jacobian_margin: float = 0.15,
    max_iter: int = 8,
) -> np.ndarray:
    return fit_penalized_family(z0, target_norm, weights, family="quadratic", jacobian_margin=jacobian_margin, max_iter=max_iter)


def run_functional_suite(n_seeds: int = 16) -> list[QuadraticFunctionalResult]:
    rows: list[QuadraticFunctionalResult] = []
    times = np.linspace(0.0, 10.0, 180)
    weights = QuadraticFunctionalWeights()
    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            rng = np.random.default_rng(9100 + 101 * seed)
            initial = scenario["builder"](rng)
            frames = simulate(initial, times, scenario["velocity"])
            final = frames[-1]

            q0, a0 = affine_moments(initial)
            z0 = np.linalg.solve(a0, (initial - q0).T).T
            qf, af = affine_moments(final)
            target_norm = np.linalg.solve(af, (final - qf).T).T

            c = fit_penalized_tensor(z0, target_norm, weights)
            rows.append(evaluate_quadratic_functional(z0, qf, af, final, c, weights, scenario["slug"], seed, "tensor_penalizado"))
    return rows


def write_outputs(base_dir: Path, rows: list[QuadraticFunctionalResult]) -> None:
    out_dir = base_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [asdict(row) for row in rows]
    (out_dir / "quadratic_functional.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    grouped: dict[str, list[QuadraticFunctionalResult]] = {}
    for row in rows:
        grouped.setdefault(row.scenario, []).append(row)

    lines = ["# Funcional Estrutural Computavel na Classe Quadratica", ""]
    lines.append("## Resultado Global")
    lines.append("")
    lines.append(
        f"- `tensor_penalizado`: total={np.mean([x.total for x in rows]):.3e}, "
        f"erro={np.mean([x.reconstruction_error for x in rows]):.3e}, "
        f"momento3={np.mean([x.third_moment_error for x in rows]):.3e}, "
        f"jac_neg={np.mean([x.negative_jacobian_fraction for x in rows]):.2%}, "
        f"norma_B={np.mean([x.tensor_norm for x in rows]):.3e}"
    )
    lines.extend(["", "## Resultado por Cenario", ""])
    for scenario, items in grouped.items():
        lines.append(f"### `{scenario}`")
        lines.append("")
        lines.append(
            f"- total={np.mean([x.total for x in items]):.3e}, erro={np.mean([x.reconstruction_error for x in items]):.3e}, "
            f"termo_rec={np.mean([x.reconstruction_term for x in items]):.3e}, "
            f"termo_m3={np.mean([x.moment3_term for x in items]):.3e}, "
            f"termo_afim={np.mean([x.affine_term for x in items]):.3e}, "
            f"termo_jac={np.mean([x.jacobian_term for x in items]):.3e}, "
            f"termo_reg={np.mean([x.regularization_term for x in items]):.3e}"
        )
        lines.append("")
    lines.extend(
        [
            "## Forma Computavel",
            "",
            "- `J_quad(B) = w_rec E_rec + w_m3 E_m3 + w_aff E_aff + w_jac E_jac + w_reg ||B||^2`.",
            "- `E_rec` usa o erro de reconstrucao em coordenadas observadas.",
            "- `E_m3` penaliza o desajuste do terceiro momento em coordenadas estruturais finais.",
            "- `E_aff` penaliza a violacao da ortogonalidade afim em vez de impô-la exatamente.",
            "- `E_jac` penaliza determinantes abaixo de uma margem positiva.",
            "- a implementacao atual usa linearizacao do termo de terceiro momento em torno de `B=0` e uma barreira jacobiana iterativa endurecida, com zona tampao e pesos crescentes para violacoes profundas.",
        ]
    )
    (out_dir / "quadratic_functional.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    rows = run_functional_suite()
    write_outputs(base_dir, rows)
    print("Funcional estrutural computavel na classe quadratica concluido.")
    print(f"Saidas em {base_dir / 'results'}")


if __name__ == "__main__":
    main()
