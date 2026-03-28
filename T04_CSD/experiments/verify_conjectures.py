import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp, trapezoid


@dataclass
class StructuralState:
    mass: float
    center: float
    scale: float
    qdot: float
    sdot: float
    kinetic_total: float
    translation_term: float
    scaling_term: float
    residual_term: float
    cross_translation_scaling: float
    cross_translation_residual: float
    cross_scaling_residual: float
    mean_constraint: float
    moment_constraint: float
    info_constraints: np.ndarray


def gaussian_eta(xi: np.ndarray, mu2: float = 1.0) -> np.ndarray:
    return np.exp(-xi**2 / (2.0 * mu2)) / np.sqrt(2.0 * np.pi * mu2)


def weighted_inner(xi: np.ndarray, eta: np.ndarray, f: np.ndarray, g: np.ndarray) -> float:
    return trapezoid(eta * f * g, xi)


def weighted_projection(
    xi: np.ndarray,
    eta: np.ndarray,
    field: np.ndarray,
    basis: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    gram = np.empty((len(basis), len(basis)))
    rhs = np.empty(len(basis))

    for i, bi in enumerate(basis):
        rhs[i] = weighted_inner(xi, eta, field, bi)
        for j, bj in enumerate(basis):
            gram[i, j] = weighted_inner(xi, eta, bi, bj)

    coeffs = np.linalg.solve(gram, rhs)
    projected = sum(c * b for c, b in zip(coeffs, basis))
    return field - projected, coeffs


def raw_transport_mode(xi: np.ndarray, t: float) -> np.ndarray:
    return (
        0.6
        - 0.35 * xi
        + 0.25 * (xi**2 - 1.0)
        + 0.18 * np.sin(2.0 * xi + 0.4 * t)
    )


def raw_information_mode(xi: np.ndarray, t: float) -> np.ndarray:
    return (
        0.4
        + 0.15 * xi
        - 0.22 * (xi**2 - 1.0)
        + 0.1 * np.cos(1.5 * xi - t)
    )


def structural_ode(t: float, y: np.ndarray) -> np.ndarray:
    mass, center, scale = y
    mass_rate = 0.05 * mass
    center_rate = 0.3 + 0.05 * np.cos(2.0 * np.pi * t)
    scale_rate = scale * (0.08 + 0.02 * np.sin(2.0 * np.pi * t))
    return np.array([mass_rate, center_rate, scale_rate])


def compute_structural_variables(x: np.ndarray, rho: np.ndarray, mu2: float = 1.0) -> tuple[float, float, float]:
    mass = trapezoid(rho, x)
    center = trapezoid(x * rho, x) / mass
    variance = trapezoid((x - center) ** 2 * rho, x) / mass
    scale = np.sqrt(variance / mu2)
    return mass, center, scale


def evaluate_state(
    t: float,
    y: np.ndarray,
    xi: np.ndarray,
    eta: np.ndarray,
    mu2: float = 1.0,
) -> StructuralState:
    mass, center, scale = y
    qdot = 0.3 + 0.05 * np.cos(2.0 * np.pi * t)
    sdot = 0.08 + 0.02 * np.sin(2.0 * np.pi * t)

    transport_basis = [np.ones_like(xi), xi]
    info_basis = [np.ones_like(xi), xi, xi**2]

    w_raw = raw_transport_mode(xi, t)
    w_perp, _ = weighted_projection(xi, eta, w_raw, transport_basis)

    psi_raw = raw_information_mode(xi, t)
    psi_perp, _ = weighted_projection(xi, eta, psi_raw, info_basis)

    x = center + scale * xi
    rho = mass * eta / scale
    v = qdot + sdot * (x - center) + scale * w_perp

    kinetic_total = trapezoid(rho * v**2, x)
    translation_term = mass * qdot**2
    scaling_term = mass * scale**2 * mu2 * sdot**2
    residual_term = mass * scale**2 * trapezoid(eta * w_perp**2, xi)

    cross_translation_scaling = 2.0 * trapezoid(rho * qdot * sdot * (x - center), x)
    cross_translation_residual = 2.0 * trapezoid(rho * qdot * scale * w_perp, x)
    cross_scaling_residual = 2.0 * trapezoid(rho * sdot * (x - center) * scale * w_perp, x)

    mean_constraint = trapezoid(eta * w_perp, xi)
    moment_constraint = trapezoid(eta * xi * w_perp, xi)
    info_constraints = np.array(
        [
            trapezoid(eta * psi_perp, xi),
            trapezoid(eta * xi * psi_perp, xi),
            trapezoid(eta * xi**2 * psi_perp, xi),
        ]
    )

    recovered_mass, recovered_center, recovered_scale = compute_structural_variables(x, rho, mu2=mu2)
    if not np.allclose([mass, center, scale], [recovered_mass, recovered_center, recovered_scale], atol=1e-4):
        raise ValueError("Structural reconstruction failed to preserve mass, center, or scale.")

    return StructuralState(
        mass=mass,
        center=center,
        scale=scale,
        qdot=qdot,
        sdot=sdot,
        kinetic_total=kinetic_total,
        translation_term=translation_term,
        scaling_term=scaling_term,
        residual_term=residual_term,
        cross_translation_scaling=cross_translation_scaling,
        cross_translation_residual=cross_translation_residual,
        cross_scaling_residual=cross_scaling_residual,
        mean_constraint=mean_constraint,
        moment_constraint=moment_constraint,
        info_constraints=info_constraints,
    )


def main() -> None:
    mu2 = 1.0
    xi = np.linspace(-8.0, 8.0, 4001)
    eta = gaussian_eta(xi, mu2=mu2)

    t_eval = np.linspace(0.0, 1.0, 101)
    initial_state = np.array([1.0, -0.25, 1.1])
    solution = solve_ivp(structural_ode, (0.0, 1.0), initial_state, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    states = [evaluate_state(t, solution.y[:, i], xi, eta, mu2=mu2) for i, t in enumerate(solution.t)]

    diagonal_error = [
        abs(s.kinetic_total - (s.translation_term + s.scaling_term + s.residual_term))
        for s in states
    ]
    cross_errors = [
        max(
            abs(s.cross_translation_scaling),
            abs(s.cross_translation_residual),
            abs(s.cross_scaling_residual),
        )
        for s in states
    ]
    transport_constraints = [max(abs(s.mean_constraint), abs(s.moment_constraint)) for s in states]
    informational_constraints = [np.max(np.abs(s.info_constraints)) for s in states]

    print("Verification summary")
    print(f"  max diagonalization error     : {max(diagonal_error):.6e}")
    print(f"  max transport cross term      : {max(cross_errors):.6e}")
    print(f"  max transport constraint err  : {max(transport_constraints):.6e}")
    print(f"  max informational constraint  : {max(informational_constraints):.6e}")
    print()
    print("Final state")
    final = states[-1]
    print(f"  mass   = {final.mass:.6f}")
    print(f"  center = {final.center:.6f}")
    print(f"  scale  = {final.scale:.6f}")
    print(f"  energy = {final.kinetic_total:.6f}")


if __name__ == "__main__":
    main()
