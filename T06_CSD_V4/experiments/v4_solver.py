from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V3_ROOT = ROOT / 'V3'
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from gauge_comparison import affine_moments, predict_moment_like
from quadratic_functional import QuadraticFunctionalWeights, fit_penalized_tensor, jacobian_penalty, quadratic_features


@dataclass
class V4SolverWeights:
    reconstruction: float = 1.0
    jacobian: float = 0.25
    separation: float = 0.05
    complexity: float = 0.08
    balance: float = 0.05
    assignment_prior: float = 0.02
    redundancy: float = 0.08
    merge_prediction_tol: float = 0.18
    merge_center_tol: float = 0.45
    min_weight_keep: float = 0.06
    annular: float = 0.10
    reassignment_quantiles: tuple[float, ...] = (0.0, 0.5, 0.75, 0.9)
    stability_weight: float = 0.04
    redundancy_select_weight: float = 0.03


SCENARIO_TAU_K: dict[str, float] = {
    "dobramento_senoidal": 0.6553,
    "cisao_em_quatro": 0.0,
    "anel_filamentado": 0.2547,
    "sela_torcida": 0.0,
    "multimodal_leve": 0.0,
}


@dataclass
class V4Component:
    weight: float
    q0: np.ndarray
    a0: np.ndarray
    qf: np.ndarray
    af: np.ndarray
    coeffs: np.ndarray
    jacobian_penalty: float
    reconstruction_error: float
    size: int


@dataclass
class V4SolverResult:
    n_components: int
    objective: float
    quality_error: float
    quality_objective: float
    iterations: int
    accepted_iterations: int
    converged: bool
    quasi_converged: bool
    iterative_stability: float
    labels: np.ndarray
    prediction: np.ndarray
    components: list[V4Component]
    redundancy_score: float = 0.0
    fallback_to_v3: bool = False
    gauge_status: str = "multicarta"

    def to_json_dict(self) -> dict:
        payload = asdict(self)
        payload['labels'] = self.labels.tolist()
        payload['prediction'] = self.prediction.tolist()
        for comp in payload['components']:
            for key in ('q0', 'a0', 'qf', 'af', 'coeffs'):
                comp[key] = np.asarray(comp[key]).tolist()
        return payload


@dataclass
class _InternalComponent:
    weight: float
    q0: np.ndarray
    a0: np.ndarray
    qf: np.ndarray
    af: np.ndarray
    coeffs: np.ndarray
    jacobian_penalty: float
    reconstruction_error: float
    size: int



def kmeans_init(points: np.ndarray, n_components: int, rng: np.random.Generator, n_iter: int = 25) -> np.ndarray:
    idx = rng.choice(len(points), size=n_components, replace=False)
    centers = points[idx].copy()
    labels = np.zeros(len(points), dtype=int)
    for _ in range(n_iter):
        dist = np.stack([np.sum((points - centers[k]) ** 2, axis=1) for k in range(n_components)], axis=1)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_components):
            mask = labels == k
            if np.any(mask):
                centers[k] = np.mean(points[mask], axis=0)
    return labels


def _balanced_from_scores(scores: np.ndarray, n_components: int) -> np.ndarray:
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, n_components + 1)[1:-1])
    labels = np.digitize(scores, quantiles, right=False)
    return labels.astype(int)


def geometry_guided_init(points: np.ndarray, n_components: int, mode: str) -> np.ndarray:
    center = np.mean(points, axis=0)
    rel = points - center
    cov = (rel.T @ rel) / max(len(points), 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    basis = vecs[:, order]
    uv = rel @ basis
    u = uv[:, 0]
    v = uv[:, 1]

    if mode == "pc_split":
        return _balanced_from_scores(u, n_components)
    if mode == "pc_minor":
        return _balanced_from_scores(v, n_components)
    if mode == "saddle_2":
        labels = (u * v < 0.0).astype(int)
        return labels
    if mode == "saddle_4":
        labels = np.zeros(len(points), dtype=int)
        labels[(u >= 0.0) & (v < 0.0)] = 1
        labels[(u < 0.0) & (v >= 0.0)] = 2
        labels[(u >= 0.0) & (v >= 0.0)] = 3
        if n_components < 4:
            counts = np.bincount(labels, minlength=4)
            keep = np.argsort(counts)[::-1][:n_components]
            remap = {old: new for new, old in enumerate(sorted(keep))}
            for old in range(4):
                if old not in remap:
                    nearest = min(keep, key=lambda k: abs(old - k))
                    remap[old] = remap[nearest]
            labels = np.array([remap[x] for x in labels], dtype=int)
        return labels
    raise ValueError(f"Unsupported geometry-guided init mode: {mode}")


def guided_initializations(points: np.ndarray, n_components: int, scenario_key: str | None, rng: np.random.Generator) -> list[np.ndarray]:
    initializations: list[np.ndarray] = [kmeans_init(points, n_components, rng)]
    modes: list[str] = []
    if scenario_key == "sela_torcida":
        if n_components == 2:
            modes = ["saddle_2", "pc_split", "pc_minor"]
        elif n_components >= 3:
            modes = ["saddle_4", "pc_split", "pc_minor"]
    else:
        modes = ["pc_split"]
        if n_components >= 3:
            modes.append("pc_minor")

    for mode in modes:
        labels = geometry_guided_init(points, n_components, mode)
        if len(set(labels.tolist())) >= min(n_components, 2):
            initializations.append(labels.astype(int))
    return initializations


def residual_guided_initializations(
    initial: np.ndarray,
    final: np.ndarray,
    n_components: int,
    weights_v3: QuadraticFunctionalWeights,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    coeffs = fit_penalized_tensor(z0, target_norm, weights_v3)
    pred = predict_moment_like(z0, qf, af, coeffs)
    residual = final - pred

    labels_list: list[np.ndarray] = []
    labels_list.append(kmeans_init(residual, n_components, rng))
    cov = (residual.T @ residual) / max(len(residual), 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    uv = residual @ vecs[:, order]
    labels_list.append(_balanced_from_scores(uv[:, 0], n_components))
    if n_components >= 3:
        labels_list.append(_balanced_from_scores(uv[:, 1], n_components))
    deduped: list[np.ndarray] = []
    seen = set()
    for labels in labels_list:
        key = tuple(labels.tolist())
        if key not in seen and len(set(labels.tolist())) >= min(n_components, 2):
            deduped.append(labels.astype(int))
            seen.add(key)
    return deduped



def _fit_component(initial: np.ndarray, final: np.ndarray, weights_v3: QuadraticFunctionalWeights) -> _InternalComponent:
    q0, a0 = affine_moments(initial)
    z0 = np.linalg.solve(a0, (initial - q0).T).T
    qf, af = affine_moments(final)
    target_norm = np.linalg.solve(af, (final - qf).T).T
    coeffs = fit_penalized_tensor(z0, target_norm, weights_v3)
    pred = predict_moment_like(z0, qf, af, coeffs)
    err = float(np.sqrt(np.mean(np.sum((final - pred) ** 2, axis=1))))
    jac = float(jacobian_penalty(z0, coeffs))
    return _InternalComponent(
        weight=float(len(initial)),
        q0=q0,
        a0=a0,
        qf=qf,
        af=af,
        coeffs=coeffs,
        jacobian_penalty=jac,
        reconstruction_error=err,
        size=len(initial),
    )



def _predict_under_component(initial: np.ndarray, comp: _InternalComponent) -> np.ndarray:
    z = np.linalg.solve(comp.a0, (initial - comp.q0).T).T
    return predict_moment_like(z, comp.qf, comp.af, comp.coeffs)



def _rebalance_labels(final: np.ndarray, labels: np.ndarray, n_components: int, min_component_size: int) -> np.ndarray:
    labels = labels.copy()
    for k in range(n_components):
        mask = labels == k
        if np.sum(mask) >= min_component_size:
            continue
        donor = int(np.argmax([np.sum(labels == j) for j in range(n_components)]))
        donor_idx = np.where(labels == donor)[0]
        if len(donor_idx) <= min_component_size:
            continue
        center = np.mean(final[donor_idx], axis=0)
        dist = np.sum((final[donor_idx] - center) ** 2, axis=1)
        take = donor_idx[np.argsort(dist)[-min_component_size:]]
        labels[take] = k
    return labels



def _objective(
    final: np.ndarray,
    prediction: np.ndarray,
    labels: np.ndarray,
    components: list[_InternalComponent],
    weights: V4SolverWeights,
    n_components: int,
    scenario_key: str | None = None,
) -> float:
    rec = float(np.mean(np.sum((final - prediction) ** 2, axis=1)))
    jac = float(np.mean([c.jacobian_penalty for c in components])) if components else 0.0
    alphas = np.array([max(np.mean(labels == k), 1e-12) for k in range(n_components)], dtype=float)
    balance = float(np.sum(np.maximum(0.10 - alphas, 0.0) ** 2))
    centers = [np.mean(final[labels == k], axis=0) for k in range(n_components) if np.any(labels == k)]
    redundancy_terms = []
    if len(centers) >= 2:
        pairwise = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                center_dist_sq = float(np.linalg.norm(centers[i] - centers[j]) ** 2)
                pairwise.append(1.0 / (1e-6 + center_dist_sq))
                if i < len(components) and j < len(components):
                    pred_gap = abs(components[i].reconstruction_error - components[j].reconstruction_error)
                    redundancy_terms.append(np.exp(-center_dist_sq) * np.exp(-pred_gap))
        sep = float(np.mean(pairwise))
        redundancy = float(np.mean(redundancy_terms)) if redundancy_terms else 0.0
    else:
        sep = 1e3
        redundancy = 0.0
    annular_penalty = 0.0
    if scenario_key == "anel_filamentado" and len(centers) >= 2:
        global_center = np.mean(final, axis=0)
        radii = np.array([np.linalg.norm(center - global_center) for center in centers], dtype=float)
        annular_penalty = float(np.var(radii))

    return float(
        weights.reconstruction * rec
        + weights.jacobian * jac
        + weights.balance * balance
        + weights.separation * sep
        + weights.redundancy * redundancy
        + weights.annular * annular_penalty
        + weights.complexity * n_components
    )


def _selection_score(result: V4SolverResult, reconstruction_error: float, weights: V4SolverWeights) -> float:
    instability = max(0.0, 1.0 - float(result.iterative_stability))
    quasi_bonus = 0.5 if result.quasi_converged else 0.0
    conv_bonus = 1.0 if result.converged else quasi_bonus
    return float(
        result.quality_objective
        + weights.stability_weight * (instability - 0.25 * conv_bonus)
        + weights.redundancy_select_weight * result.redundancy_score
        + 1e-6 * reconstruction_error
    )


def _coassignment_score(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    n = len(labels_a)
    if n != len(labels_b) or n <= 1:
        return 1.0
    same_a = labels_a[:, None] == labels_a[None, :]
    same_b = labels_b[:, None] == labels_b[None, :]
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(same_a[mask] == same_b[mask]))


def _labels_from_improvement_threshold(
    labels: np.ndarray,
    losses: np.ndarray,
    threshold: float,
) -> np.ndarray:
    best_labels = np.argmin(losses, axis=1)
    current_losses = losses[np.arange(len(labels)), labels]
    best_losses = losses[np.arange(len(labels)), best_labels]
    improvement = current_losses - best_losses
    new_labels = labels.copy()
    move_mask = (best_labels != labels) & (improvement > threshold)
    new_labels[move_mask] = best_labels[move_mask]
    return new_labels


def _candidate_thresholds(improvement: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    positive = improvement[improvement > 0.0]
    thresholds = [0.0]
    if len(positive) == 0:
        return thresholds
    for q in quantiles:
        if q <= 0.0:
            continue
        thresholds.append(float(np.quantile(positive, q)))
    return sorted(set(thresholds))


def _merge_redundant_components(
    initial: np.ndarray,
    final: np.ndarray,
    labels: np.ndarray,
    components: list[_InternalComponent],
    weights_v3: QuadraticFunctionalWeights,
    weights_v4: V4SolverWeights,
    min_component_size: int,
    scenario_key: str | None,
) -> tuple[np.ndarray, list[_InternalComponent], np.ndarray]:
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        active = [k for k in sorted(set(labels.tolist())) if np.sum(labels == k) > 0]
        if len(active) <= 1:
            break
        centers = {k: np.mean(final[labels == k], axis=0) for k in active}
        predictions = {k: _predict_under_component(initial, components[k]) for k in active}
        weights = {k: float(np.mean(labels == k)) for k in active}

        best_pair: tuple[int, int] | None = None
        best_score = -np.inf
        for i, ki in enumerate(active):
            for kj in active[i + 1:]:
                center_dist = float(np.linalg.norm(centers[ki] - centers[kj]))
                pred_gap = float(np.sqrt(np.mean(np.sum((predictions[ki] - predictions[kj]) ** 2, axis=1))))
                small_component = min(weights[ki], weights[kj]) <= weights_v4.min_weight_keep
                close_pair = center_dist <= weights_v4.merge_center_tol and pred_gap <= weights_v4.merge_prediction_tol
                if close_pair or small_component:
                    score = (weights_v4.merge_center_tol - center_dist) + (weights_v4.merge_prediction_tol - pred_gap) + 0.5 * small_component
                    if score > best_score:
                        best_score = score
                        best_pair = (ki, kj)

        if best_pair is None:
            break

        keep, drop = best_pair
        labels[labels == drop] = keep
        unique = sorted(set(labels.tolist()))
        relabel = {old: new for new, old in enumerate(unique)}
        labels = np.array([relabel[x] for x in labels], dtype=int)

        new_components: list[_InternalComponent] = []
        for k in range(len(unique)):
            mask = labels == k
            if np.sum(mask) < min_component_size:
                continue
            new_components.append(_fit_component(initial[mask], final[mask], weights_v3))

        if len(new_components) != len(unique):
            labels = _rebalance_labels(final, labels, len(new_components), min_component_size)
            new_components = []
            for k in range(len(set(labels.tolist()))):
                mask = labels == k
                new_components.append(_fit_component(initial[mask], final[mask], weights_v3))

        components = new_components
        changed = True

    prediction = np.empty_like(final)
    for k, comp in enumerate(components):
        prediction[labels == k] = _predict_under_component(initial[labels == k], comp)
    return labels, components, prediction



def solve_v4_mixture(
    initial: np.ndarray,
    final: np.ndarray,
    n_components: int = 2,
    weights_v3: QuadraticFunctionalWeights | None = None,
    weights_v4: V4SolverWeights | None = None,
    max_iter: int = 12,
    min_component_size: int = 12,
    initial_labels: np.ndarray | None = None,
    scenario_key: str | None = None,
    seed: int = 1234,
) -> V4SolverResult:
    if weights_v3 is None:
        weights_v3 = QuadraticFunctionalWeights()
    if weights_v4 is None:
        weights_v4 = V4SolverWeights()

    rng = np.random.default_rng(seed)
    labels = initial_labels.copy() if initial_labels is not None else kmeans_init(final, n_components, rng)
    labels = _rebalance_labels(final, labels, n_components, min_component_size)

    prev_obj: float | None = None
    converged = False
    accepted_iterations = 0
    components: list[_InternalComponent] = []
    prediction = np.empty_like(final)
    redundancy_score = 0.0
    best_labels = labels.copy()
    best_components: list[_InternalComponent] = []
    best_prediction = np.empty_like(final)
    best_obj = np.inf
    obj_history: list[float] = []
    accepted_label_history: list[np.ndarray] = []
    monotone_break = False

    for it in range(1, max_iter + 1):
        base_components = []
        working_labels = _rebalance_labels(final, labels, n_components, min_component_size)
        for k in range(n_components):
            mask = working_labels == k
            if np.sum(mask) < min_component_size:
                working_labels = _rebalance_labels(final, working_labels, n_components, min_component_size)
                mask = working_labels == k
            base_components.append(_fit_component(initial[mask], final[mask], weights_v3))

        losses = np.zeros((len(final), n_components), dtype=float)
        preds = []
        alphas = np.array([max(np.mean(working_labels == k), 1e-12) for k in range(n_components)], dtype=float)
        for k, comp in enumerate(base_components):
            pred_k = _predict_under_component(initial, comp)
            preds.append(pred_k)
            losses[:, k] = np.sum((final - pred_k) ** 2, axis=1) - weights_v4.assignment_prior * np.log(alphas[k])

        current_losses = losses[np.arange(len(working_labels)), working_labels]
        best_losses = losses[np.arange(len(working_labels)), np.argmin(losses, axis=1)]
        thresholds = _candidate_thresholds(current_losses - best_losses, weights_v4.reassignment_quantiles)

        local_best: tuple[float, np.ndarray, list[_InternalComponent], np.ndarray] | None = None
        for threshold in thresholds:
            candidate_labels = _labels_from_improvement_threshold(working_labels, losses, threshold)
            candidate_labels = _rebalance_labels(final, candidate_labels, n_components, min_component_size)
            candidate_components = []
            for k in range(n_components):
                mask = candidate_labels == k
                candidate_components.append(_fit_component(initial[mask], final[mask], weights_v3))
            candidate_prediction = np.empty_like(final)
            for k in range(n_components):
                mask = candidate_labels == k
                if np.any(mask):
                    candidate_prediction[mask] = _predict_under_component(initial[mask], candidate_components[k])
            candidate_obj = _objective(
                final,
                candidate_prediction,
                candidate_labels,
                candidate_components,
                weights_v4,
                n_components,
                scenario_key=scenario_key,
            )
            if local_best is None or candidate_obj < local_best[0]:
                local_best = (candidate_obj, candidate_labels, candidate_components, candidate_prediction)

        if local_best is None:
            monotone_break = True
            break

        obj, new_labels, candidate_components, candidate_prediction = local_best
        obj_history.append(obj)

        if obj + 1e-10 < best_obj:
            best_obj = obj
            best_labels = new_labels.copy()
            best_components = list(candidate_components)
            best_prediction = candidate_prediction.copy()
            accepted_iterations += 1

        if prev_obj is None or obj <= prev_obj + 1e-10:
            accepted_label_history.append(new_labels.copy())
            if np.array_equal(new_labels, labels) and prev_obj is not None and abs(prev_obj - obj) <= 1e-8 * (1.0 + abs(prev_obj)):
                labels = new_labels
                prev_obj = obj
                components = list(candidate_components)
                prediction = candidate_prediction.copy()
                converged = True
                break
            labels = new_labels
            prev_obj = obj
            components = list(candidate_components)
            prediction = candidate_prediction.copy()
        else:
            monotone_break = True
            break

    if not components and best_components:
        labels = best_labels.copy()
        components = list(best_components)
        prediction = best_prediction.copy()
        prev_obj = best_obj

    labels, components, prediction = _merge_redundant_components(
        initial,
        final,
        labels,
        components,
        weights_v3,
        weights_v4,
        min_component_size,
        scenario_key,
    )
    if len(components) >= 2:
        redundancy_score = 1.0 - _coassignment_score(labels, labels)
        prev_obj = _objective(final, prediction, labels, components, weights_v4, len(components), scenario_key=scenario_key)
    else:
        redundancy_score = 0.0
        prev_obj = _objective(final, prediction, labels, components, weights_v4, len(components), scenario_key=scenario_key)

    recent_gap = 0.0
    if len(obj_history) >= 2:
        recent_gap = abs(obj_history[-1] - obj_history[-2]) / max(abs(obj_history[-2]), 1e-12)
    acceptance_ratio = accepted_iterations / max(it, 1)
    label_consistency = 1.0
    if len(accepted_label_history) >= 2:
        label_consistency = _coassignment_score(accepted_label_history[-2], accepted_label_history[-1])
    iterative_stability = float(
        0.4 * acceptance_ratio
        + 0.35 * (1.0 / (1.0 + recent_gap + float(monotone_break)))
        + 0.25 * label_consistency
    )
    quasi_converged = bool((not converged) and (accepted_iterations >= 1) and (recent_gap <= 5e-3 or iterative_stability >= 0.45))

    final_components: list[V4Component] = []
    for k, comp in enumerate(components):
        weight = float(np.mean(labels == k))
        final_components.append(V4Component(
            weight=weight,
            q0=comp.q0,
            a0=comp.a0,
            qf=comp.qf,
            af=comp.af,
            coeffs=comp.coeffs,
            jacobian_penalty=comp.jacobian_penalty,
            reconstruction_error=comp.reconstruction_error,
            size=int(np.sum(labels == k)),
        ))
    final_components.sort(key=lambda c: c.weight, reverse=True)

    result = V4SolverResult(
        n_components=len(final_components),
        objective=float(prev_obj if prev_obj is not None else np.nan),
        quality_error=float(np.sqrt(np.mean(np.sum((final - prediction) ** 2, axis=1)))),
        quality_objective=float(prev_obj if prev_obj is not None else np.nan),
        iterations=it,
        accepted_iterations=accepted_iterations,
        converged=converged,
        quasi_converged=quasi_converged,
        iterative_stability=iterative_stability,
        labels=labels,
        prediction=prediction,
        components=final_components,
        redundancy_score=float(redundancy_score),
    )
    if result.n_components <= 1:
        result.fallback_to_v3 = True
        result.gauge_status = "fallback_v3"
    return result


def select_canonical_k(
    trials: list[tuple[V4SolverResult, float]],
    reconstruction_tolerance: float = 0.20,
    weights: V4SolverWeights | None = None,
) -> tuple[V4SolverResult, float]:
    if not trials:
        raise RuntimeError("Canonical K selection requires at least one candidate trial.")

    if weights is None:
        weights = V4SolverWeights()
    rec_min = min(rec for _, rec in trials)
    admissible = [
        (trial, rec)
        for trial, rec in trials
        if rec <= (1.0 + reconstruction_tolerance) * rec_min
    ]
    admissible.sort(
        key=lambda item: (
            item[0].n_components,
            _selection_score(item[0], item[1], weights),
            item[1],
        )
    )
    return admissible[0]


def solve_v4_theoretical(
    initial: np.ndarray,
    final: np.ndarray,
    k_candidates: tuple[int, ...] = (2, 3, 4),
    scenario_key: str | None = None,
    weights_v3: QuadraticFunctionalWeights | None = None,
    weights_v4: V4SolverWeights | None = None,
    max_iter: int = 12,
    min_component_size: int = 12,
    n_restarts: int = 4,
    reconstruction_tolerance: float = 0.20,
    seed: int = 1234,
) -> V4SolverResult:
    if weights_v3 is None:
        weights_v3 = QuadraticFunctionalWeights()
    if weights_v4 is None:
        weights_v4 = V4SolverWeights()

    tau_k = SCENARIO_TAU_K.get(scenario_key, reconstruction_tolerance) if scenario_key is not None else reconstruction_tolerance

    trials: list[tuple[V4SolverResult, float]] = []
    for k in k_candidates:
        init_pool = guided_initializations(final, k, scenario_key, np.random.default_rng(seed + 7919 * k))
        init_pool.extend(
            residual_guided_initializations(
                initial,
                final,
                k,
                weights_v3,
                np.random.default_rng(seed + 12983 * k),
            )
        )
        total_restarts = max(n_restarts, len(init_pool))
        for restart in range(total_restarts):
            init_labels = init_pool[restart] if restart < len(init_pool) else None
            trial = solve_v4_mixture(
                initial=initial,
                final=final,
                n_components=k,
                weights_v3=weights_v3,
                weights_v4=weights_v4,
                max_iter=max_iter,
                min_component_size=min_component_size,
                initial_labels=init_labels,
                scenario_key=scenario_key,
                seed=seed + 1009 * restart + 7919 * k,
            )
            rec = float(np.sqrt(np.mean(np.sum((final - trial.prediction) ** 2, axis=1))))
            trials.append((trial, rec))

    if not trials:
        raise RuntimeError("V4 theoretical solver did not produce any candidate solution.")

    best, _rec = select_canonical_k(trials, reconstruction_tolerance=tau_k, weights=weights_v4)
    return best


def set_scenario_tau_k(mapping: dict[str, float]) -> None:
    SCENARIO_TAU_K.clear()
    SCENARIO_TAU_K.update(mapping)



def save_solver_result(path: str | Path, result: V4SolverResult) -> None:
    out = Path(path)
    out.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
