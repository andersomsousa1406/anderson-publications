import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ROOT = None
for parent in SCRIPT_PATH.parents:
    if (parent / "publications").exists():
        ROOT = parent
        break
if ROOT is None:
    raise RuntimeError("Could not locate the repository root.")

import matplotlib

if "--save" in sys.argv or "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SEED = 52
TRUCK_COUNT = 6
POINT_COUNT = 120
CAPACITY_TARGET = POINT_COUNT // TRUCK_COUNT
DEPOT = np.array([0.0, 0.0], dtype=float)


def generate_delivery_points(seed: int = SEED, point_count: int = POINT_COUNT) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hub_centers = np.array(
        [
            [-10.0, 10.5],
            [11.5, 8.0],
            [-12.0, -6.0],
            [8.5, -10.5],
            [1.5, 13.0],
            [13.0, 1.0],
        ],
        dtype=float,
    )
    counts = np.full(len(hub_centers), point_count // len(hub_centers), dtype=int)
    counts[: point_count - int(np.sum(counts))] += 1

    points = []
    for center, count in zip(hub_centers, counts):
        cloud = center + rng.normal(scale=[2.4, 2.0], size=(count, 2))
        points.append(cloud)
    points = np.vstack(points)
    rng.shuffle(points)
    return points


def route_length(points: np.ndarray, route: list[int], depot: np.ndarray = DEPOT) -> float:
    if not route:
        return 0.0
    total = float(np.linalg.norm(points[route[0]] - depot))
    for a, b in zip(route[:-1], route[1:]):
        total += float(np.linalg.norm(points[a] - points[b]))
    total += float(np.linalg.norm(points[route[-1]] - depot))
    return total


def total_length(points: np.ndarray, routes: list[list[int]]) -> float:
    return float(sum(route_length(points, route) for route in routes))


def baseline_routes(points: np.ndarray, truck_count: int = TRUCK_COUNT) -> list[list[int]]:
    centered = points - DEPOT
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    order = np.argsort(angles)
    splits = np.array_split(order, truck_count)
    routes = []
    for split in splits:
        route = list(split.tolist())
        route.sort(key=lambda idx: float(np.linalg.norm(points[idx] - DEPOT)))
        routes.append(route)
    return routes


def initialize_centroids(points: np.ndarray, truck_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(0, len(points)))]
    while len(chosen) < truck_count:
        d2 = np.full(len(points), np.inf)
        for idx in chosen:
            diff = points - points[idx]
            d2 = np.minimum(d2, np.sum(diff * diff, axis=1))
        probs = d2 / max(float(np.sum(d2)), 1e-12)
        chosen.append(int(rng.choice(len(points), p=probs)))
    return points[chosen].copy()


def capacity_aware_assignment(points: np.ndarray, centroids: np.ndarray, capacity_target: int) -> np.ndarray:
    counts = np.zeros(len(centroids), dtype=int)
    assignment = np.full(len(points), -1, dtype=int)
    point_order = np.argsort(np.linalg.norm(points - np.mean(points, axis=0), axis=1))[::-1]

    for idx in point_order:
        dists = np.linalg.norm(points[idx] - centroids, axis=1)
        penalties = np.maximum(counts - capacity_target + 1, 0) * 1.8
        cluster = int(np.argmin(dists + penalties))
        assignment[idx] = cluster
        counts[cluster] += 1
    return assignment


def rebalance_assignment(points: np.ndarray, assignment: np.ndarray, truck_count: int, capacity_target: int) -> np.ndarray:
    assignment = assignment.copy()
    while True:
        counts = np.bincount(assignment, minlength=truck_count)
        oversized = np.where(counts > capacity_target + 1)[0]
        undersized = np.where(counts < capacity_target - 1)[0]
        if len(oversized) == 0 or len(undersized) == 0:
            break
        moved = False
        for src in oversized:
            src_idx = np.where(assignment == src)[0]
            src_points = points[src_idx]
            src_center = np.mean(src_points, axis=0)
            for idx in src_idx:
                dest_choices = []
                for dest in undersized:
                    dest_points = points[assignment == dest]
                    dest_center = np.mean(dest_points, axis=0) if len(dest_points) > 0 else src_center
                    gain = np.linalg.norm(points[idx] - src_center) - np.linalg.norm(points[idx] - dest_center)
                    dest_choices.append((gain, int(dest)))
                if not dest_choices:
                    continue
                _, best_dest = max(dest_choices, key=lambda item: item[0])
                assignment[idx] = best_dest
                moved = True
                break
            if moved:
                break
        if not moved:
            break
    return assignment


def build_structural_clusters(points: np.ndarray, truck_count: int, capacity_target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    centroids = initialize_centroids(points, truck_count, seed)
    assignment = np.zeros(len(points), dtype=int)
    for _ in range(8):
        assignment = capacity_aware_assignment(points, centroids, capacity_target)
        assignment = rebalance_assignment(points, assignment, truck_count, capacity_target)
        new_centroids = centroids.copy()
        for k in range(truck_count):
            mask = assignment == k
            if np.any(mask):
                new_centroids[k] = np.mean(points[mask], axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-3):
            centroids = new_centroids
            break
        centroids = new_centroids
    return assignment, centroids


def nearest_neighbor_route(points: np.ndarray, indices: list[int], depot: np.ndarray = DEPOT) -> list[int]:
    remaining = set(indices)
    current = depot
    route: list[int] = []
    while remaining:
        next_idx = min(remaining, key=lambda idx: float(np.linalg.norm(points[idx] - current)))
        route.append(next_idx)
        remaining.remove(next_idx)
        current = points[next_idx]
    return route


def projected_order_route(points: np.ndarray, indices: list[int], centroid: np.ndarray) -> list[int]:
    if len(indices) <= 2:
        return list(indices)
    subset = points[indices]
    centered = subset - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    scores = centered @ axis
    order = np.argsort(scores)
    return [indices[i] for i in order]


def angular_route(points: np.ndarray, indices: list[int], centroid: np.ndarray) -> list[int]:
    subset = points[indices] - centroid
    order = np.argsort(np.arctan2(subset[:, 1], subset[:, 0]))
    return [indices[i] for i in order]


def best_initial_route(points: np.ndarray, indices: list[int], centroid: np.ndarray) -> list[int]:
    candidates = [
        nearest_neighbor_route(points, indices),
        projected_order_route(points, indices, centroid),
        angular_route(points, indices, centroid),
    ]
    best = min(candidates, key=lambda route: route_length(points, route))
    return best


def two_opt(points: np.ndarray, route: list[int], max_passes: int = 2) -> list[int]:
    best = route[:]
    for _ in range(max_passes):
        improved = False
        improved = False
        best_length = route_length(points, best)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                candidate = best[:]
                candidate[i:j] = reversed(candidate[i:j])
                cand_length = route_length(points, candidate)
                if cand_length + 1e-9 < best_length:
                    best = candidate
                    improved = True
                    best_length = cand_length
                    break
            if improved:
                break
        route = best[:]
        if not improved:
            break
    return best


def relocate_between_routes(points: np.ndarray, routes: list[list[int]], capacity_target: int) -> tuple[list[list[int]], bool]:
    best_routes = [route[:] for route in routes]
    best_total = total_length(points, best_routes)
    improved = False
    route_lengths = np.asarray([route_length(points, route) for route in routes], dtype=float)
    src_candidates = np.argsort(route_lengths)[::-1][: min(3, len(routes))]
    dst_candidates = np.argsort(route_lengths)[: min(4, len(routes))]
    for src_idx in src_candidates:
        src = routes[int(src_idx)]
        src_positions = [0, len(src) // 2, len(src) - 1] if src else []
        for pos, customer in enumerate(src):
            if pos not in src_positions:
                continue
            for dst_idx in dst_candidates:
                dst = routes[int(dst_idx)]
                if src_idx == dst_idx:
                    continue
                if len(dst) >= capacity_target + 2:
                    continue
                candidate_insert_positions = [0, len(dst) // 2, len(dst)] if dst else [0]
                for insert_pos in candidate_insert_positions:
                    candidate_routes = [route[:] for route in routes]
                    candidate_routes[src_idx].pop(pos)
                    candidate_routes[dst_idx].insert(insert_pos, customer)
                    candidate_routes[src_idx] = two_opt(points, candidate_routes[src_idx], max_passes=1)
                    candidate_routes[dst_idx] = two_opt(points, candidate_routes[dst_idx], max_passes=1)
                    score = total_length(points, candidate_routes)
                    if score + 1e-9 < best_total:
                        best_total = score
                        best_routes = candidate_routes
                        improved = True
    return best_routes, improved


def gonm_routes(points: np.ndarray, truck_count: int = TRUCK_COUNT, capacity_target: int = CAPACITY_TARGET, seed: int = SEED) -> tuple[list[list[int]], list[float], np.ndarray]:
    assignment, centroids = build_structural_clusters(points, truck_count, capacity_target, seed)
    routes = []
    for k in range(truck_count):
        indices = np.where(assignment == k)[0].tolist()
        routes.append(best_initial_route(points, indices, centroids[k]))

    trace = [total_length(points, routes)]

    for _ in range(3):
        routes = [two_opt(points, route, max_passes=1) for route in routes]
        trace.append(total_length(points, routes))
        routes, improved = relocate_between_routes(points, routes, capacity_target)
        trace.append(total_length(points, routes))
        if not improved:
            break

    routes = [two_opt(points, route, max_passes=2) for route in routes]
    trace.append(total_length(points, routes))
    return routes, trace, centroids


def draw_routes(ax, points: np.ndarray, routes: list[list[int]], title: str, depot: np.ndarray = DEPOT) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, len(routes)))
    ax.set_title(title)
    ax.scatter(points[:, 0], points[:, 1], s=16, color="#64748b", alpha=0.55)
    ax.scatter([depot[0]], [depot[1]], marker="*", s=240, color="#dc2626", edgecolors="black", linewidths=0.6)
    for color, route in zip(colors, routes):
        if not route:
            continue
        path = np.vstack([depot, points[route], depot])
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.8, alpha=0.95)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")


def render_result(points: np.ndarray, baseline: list[list[int]], gonm: list[list[int]], frontier_stats: dict, trace: list[float]) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.2])
    ax_base = fig.add_subplot(gs[0, 0])
    ax_gonm = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    draw_routes(ax_base, points, baseline, "Baseline geometrico por sweep angular")
    draw_routes(ax_gonm, points, gonm, "Rotas GONM: estrutural + contrativa + 2-opt")

    ax_trace.plot(np.asarray(trace, dtype=float), color="#2563eb", lw=2.0)
    ax_trace.set_title("Contracao do comprimento total")
    ax_trace.set_xlabel("etapa agregada")
    ax_trace.set_ylabel("comprimento total")
    ax_trace.grid(True, linestyle=":", alpha=0.5)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | logistics and geometric VRP",
                "",
                f"pontos de entrega = {frontier_stats['point_count']}",
                f"caminhoes = {frontier_stats['truck_count']}",
                f"capacidade alvo = {frontier_stats['capacity_target']}",
                "",
                f"baseline total = {frontier_stats['baseline_total']:.2f}",
                f"GONM total = {frontier_stats['gonm_total']:.2f}",
                f"ganho absoluto = {frontier_stats['gain_abs']:.2f}",
                f"ganho relativo = {frontier_stats['gain_pct']:.2f}%",
                "",
                f"maior rota baseline = {frontier_stats['baseline_max']:.2f}",
                f"maior rota GONM = {frontier_stats['gonm_max']:.2f}",
                "",
                "leitura:",
                "CSD organiza zonas de entrega,",
                "CPP contrai a ordem local,",
                "e o fechamento final faz trocas 2-opt.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.8,
        family="monospace",
    )

    fig.suptitle("GONM | Logistica e roteamento geometrico de entregas", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(payload: dict, fig: plt.Figure) -> tuple[Path, Path, Path]:
    out_dir = ROOT / "publications" / "T07_GONM" / "results" / "gonm_logistics_vrp"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    image_path = out_dir / "gonm_logistics_vrp.png"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# GONM | Geometric VRP",
                "",
                f"- delivery points: `{payload['point_count']}`",
                f"- trucks: `{payload['truck_count']}`",
                f"- baseline total length: `{payload['baseline_total']:.2f}`",
                f"- GONM total length: `{payload['gonm_total']:.2f}`",
                f"- relative gain: `{payload['gain_pct']:.2f}%`",
                "",
                "## Files",
                "",
                "- image: `publications/T07_GONM/results/gonm_logistics_vrp/gonm_logistics_vrp.png`",
                "- summary: `publications/T07_GONM/results/gonm_logistics_vrp/summary.json`",
            ]
        ),
        encoding="utf-8",
    )
    fig.savefig(image_path, dpi=170)
    return json_path, md_path, image_path


def main() -> None:
    points = generate_delivery_points()
    baseline = baseline_routes(points)
    gonm, trace, centroids = gonm_routes(points)

    baseline_lengths = [route_length(points, route) for route in baseline]
    gonm_lengths = [route_length(points, route) for route in gonm]
    baseline_total = float(np.sum(baseline_lengths))
    gonm_total = float(np.sum(gonm_lengths))

    payload = {
        "seed": SEED,
        "point_count": len(points),
        "truck_count": TRUCK_COUNT,
        "capacity_target": CAPACITY_TARGET,
        "baseline_total": baseline_total,
        "gonm_total": gonm_total,
        "gain_abs": baseline_total - gonm_total,
        "gain_pct": 100.0 * (baseline_total - gonm_total) / baseline_total,
        "baseline_route_lengths": baseline_lengths,
        "gonm_route_lengths": gonm_lengths,
        "trace_total_length": [float(v) for v in trace],
        "depot": DEPOT.tolist(),
        "points": points.tolist(),
        "baseline_routes": baseline,
        "gonm_routes": gonm,
        "centroids": centroids.tolist(),
    }

    fig = render_result(
        points,
        baseline,
        gonm,
        {
            "point_count": len(points),
            "truck_count": TRUCK_COUNT,
            "capacity_target": CAPACITY_TARGET,
            "baseline_total": baseline_total,
            "gonm_total": gonm_total,
            "gain_abs": baseline_total - gonm_total,
            "gain_pct": 100.0 * (baseline_total - gonm_total) / baseline_total,
            "baseline_max": float(np.max(baseline_lengths)),
            "gonm_max": float(np.max(gonm_lengths)),
        },
        trace,
    )
    json_path, md_path, image_path = save_outputs(payload, fig)
    print(f"Resumo salvo em {json_path}")
    print(f"Resumo legivel salvo em {md_path}")
    print(f"Imagem salva em {image_path}")
    if "--no-show" in sys.argv:
        plt.close(fig)
    elif "--save" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
