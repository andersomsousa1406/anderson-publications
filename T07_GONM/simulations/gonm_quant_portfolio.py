import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ROOT = None
LIBRARY_ROOT = None
for parent in SCRIPT_PATH.parents:
    candidate_library = parent / "publications" / "library"
    if candidate_library.exists():
        LIBRARY_ROOT = candidate_library
        ROOT = parent
        break
if LIBRARY_ROOT is None or ROOT is None:
    raise RuntimeError("Could not locate publications/library for the anderson package.")
if str(LIBRARY_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBRARY_ROOT))

import matplotlib

if "--save" in sys.argv or "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from anderson.noise import CoolingNoise
from anderson.optimizers.gonm import GONMConfig, GONMOptimizer
from anderson.problems import SparsePortfolioProblem


SEED = 41
ASSET_COUNT = 60
SECTOR_NAMES = ["Tech", "Health", "Energy", "Finance", "Industry", "Consumer"]


def build_market(seed: int = SEED) -> SparsePortfolioProblem:
    rng = np.random.default_rng(seed)
    sector_count = len(SECTOR_NAMES)
    assets_per_sector = ASSET_COUNT // sector_count
    sectors = np.repeat(np.arange(sector_count), assets_per_sector)
    sector_factor_returns = np.asarray([0.18, 0.14, 0.11, 0.13, 0.12, 0.10], dtype=float)
    sector_factor_vols = np.asarray([0.26, 0.18, 0.24, 0.20, 0.19, 0.17], dtype=float)

    expected_returns = []
    asset_names = []
    loadings = np.zeros((ASSET_COUNT, sector_count), dtype=float)
    idio_var = np.zeros(ASSET_COUNT, dtype=float)
    for idx in range(ASSET_COUNT):
        sector = int(sectors[idx])
        local_id = idx - sector * assets_per_sector + 1
        asset_names.append(f"{SECTOR_NAMES[sector][:3].upper()}{local_id:02d}")
        base_return = sector_factor_returns[sector]
        expected_returns.append(base_return + rng.normal(scale=0.018))
        loadings[idx, sector] = 0.90 + 0.18 * rng.random()
        if sector > 0:
            loadings[idx, sector - 1] += 0.05 * rng.random()
        if sector < sector_count - 1:
            loadings[idx, sector + 1] += 0.05 * rng.random()
        idio_var[idx] = (0.06 + 0.04 * rng.random()) ** 2

    factor_cov = np.diag(sector_factor_vols**2)
    covariance = loadings @ factor_cov @ loadings.T + np.diag(idio_var)
    covariance += 0.015 * np.ones_like(covariance) / ASSET_COUNT

    return SparsePortfolioProblem(
        expected_returns=np.asarray(expected_returns, dtype=float),
        covariance=covariance,
        asset_names=asset_names,
        sector_names=SECTOR_NAMES,
        sectors=sectors,
        risk_aversion=7.5,
        return_weight=1.0,
        sparsity_weight=0.10,
        sparsity_epsilon=1e-4,
        sector_cap=0.42,
    )


def make_optimizer(problem: SparsePortfolioProblem) -> GONMOptimizer:
    return GONMOptimizer(
        problem=problem,
        noise=CoolingNoise(start_std=0.010, mid_std=0.004, late_std=0.001, final_std=0.0),
        config=GONMConfig(
            population=96,
            phase1_keep=12,
            phase2_steps=220,
            phase3_steps=96,
            box_scale=1.2,
            thermal_scale=0.020,
            momentum=0.88,
            step_size=0.050,
            terminal_radii=(0.085, 0.050, 0.025, 0.012),
        ),
    )


def projected_gradient_baseline(problem: SparsePortfolioProblem, steps: int = 320, lr: float = 0.08) -> np.ndarray:
    w = np.full(problem.dimensions, 1.0 / problem.dimensions, dtype=float)
    best_w = w.copy()
    best_e = problem.energy(w)
    for _ in range(steps):
        grad = problem.gradient(w)
        w = problem.project(w - lr * grad)
        energy = problem.energy(w)
        if energy < best_e:
            best_e = energy
            best_w = w.copy()
    return best_w


def random_frontier(problem: SparsePortfolioProblem, rng: np.random.Generator, samples: int = 1800) -> np.ndarray:
    rows = []
    for _ in range(samples):
        w = problem.random_geometry(rng, scale=0.9)
        rows.append(
            [
                np.sqrt(problem.portfolio_variance(w)),
                problem.expected_portfolio_return(w),
                problem.energy(w),
                problem.effective_assets(w),
            ]
        )
    return np.asarray(rows, dtype=float)


def top_positions(problem: SparsePortfolioProblem, w: np.ndarray, top_n: int = 12) -> list[tuple[str, float]]:
    projected = problem.project(w)
    order = np.argsort(projected)[::-1][:top_n]
    return [(problem.asset_names[i], float(projected[i])) for i in order]


def summarize_portfolio(problem: SparsePortfolioProblem, w: np.ndarray) -> dict:
    projected = problem.project(w)
    return {
        "weights": projected.tolist(),
        "objective": float(problem.energy(projected)),
        "risk_std": float(np.sqrt(problem.portfolio_variance(projected))),
        "expected_return": float(problem.expected_portfolio_return(projected)),
        "effective_assets_2pct": int(problem.effective_assets(projected, threshold=0.02)),
        "effective_assets_1pct": int(problem.effective_assets(projected, threshold=0.01)),
        "sector_allocations": problem.sector_allocations(projected).tolist(),
        "top_positions": top_positions(problem, projected),
    }


def render_result(problem: SparsePortfolioProblem, frontier: np.ndarray, baseline: dict, gonm: dict, trace: list[float]) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.25])
    ax_frontier = fig.add_subplot(gs[0, 0])
    ax_weights = fig.add_subplot(gs[0, 1])
    ax_sector = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_frontier.scatter(frontier[:, 0], frontier[:, 1], s=12, alpha=0.18, color="#94a3b8", label="Random feasible")
    ax_frontier.scatter([baseline["risk_std"]], [baseline["expected_return"]], color="#ef4444", s=90, label="Projected gradient")
    ax_frontier.scatter([gonm["risk_std"]], [gonm["expected_return"]], color="#2563eb", s=90, label="GONM")
    ax_frontier.set_title("Risco vs retorno em carteiras viaveis")
    ax_frontier.set_xlabel("risco (desvio padrao)")
    ax_frontier.set_ylabel("retorno esperado")
    ax_frontier.grid(True, linestyle=":", alpha=0.5)
    ax_frontier.legend(loc="best")

    baseline_top = baseline["top_positions"]
    gonm_top = gonm["top_positions"]
    labels = [name for name, _ in gonm_top]
    baseline_map = dict(baseline_top)
    gonm_map = dict(gonm_top)
    x = np.arange(len(labels))
    width = 0.38
    ax_weights.bar(x - width / 2, [baseline_map.get(label, 0.0) for label in labels], width=width, color="#f87171", label="Baseline")
    ax_weights.bar(x + width / 2, [gonm_map.get(label, 0.0) for label in labels], width=width, color="#60a5fa", label="GONM")
    ax_weights.set_title("Principais pesos da carteira")
    ax_weights.set_xticks(x)
    ax_weights.set_xticklabels(labels, rotation=45, ha="right")
    ax_weights.set_ylabel("peso")
    ax_weights.legend(loc="best")
    ax_weights.grid(True, axis="y", linestyle=":", alpha=0.4)

    sectors_x = np.arange(len(problem.sector_names))
    ax_sector.bar(sectors_x - width / 2, baseline["sector_allocations"], width=width, color="#f87171", label="Baseline")
    ax_sector.bar(sectors_x + width / 2, gonm["sector_allocations"], width=width, color="#60a5fa", label="GONM")
    ax_sector.set_title("Alocacao por setor")
    ax_sector.set_xticks(sectors_x)
    ax_sector.set_xticklabels(problem.sector_names, rotation=20)
    ax_sector.set_ylabel("peso agregado")
    ax_sector.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax_sector.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM | sparse portfolio optimization",
                "",
                f"ativos = {problem.dimensions}",
                f"setores = {len(problem.sector_names)}",
                f"objetivo GONM = {gonm['objective']:.6f}",
                f"objetivo baseline = {baseline['objective']:.6f}",
                "",
                f"risco GONM = {gonm['risk_std']:.4f}",
                f"retorno GONM = {gonm['expected_return']:.4f}",
                f"ativos efetivos >=2% = {gonm['effective_assets_2pct']}",
                "",
                f"risco baseline = {baseline['risk_std']:.4f}",
                f"retorno baseline = {baseline['expected_return']:.4f}",
                f"ativos efetivos >=2% = {baseline['effective_assets_2pct']}",
                "",
                f"melhor energia GONM final = {trace[-1]:.6f}",
                f"energia inicial GONM = {trace[0]:.6f}",
                "",
                "interpretacao:",
                "GONM busca uma carteira mais compacta",
                "sem perder a disciplina risco-retorno.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.8,
        family="monospace",
    )

    inset = ax_text.inset_axes([0.05, 0.02, 0.9, 0.24])
    inset.plot(np.asarray(trace, dtype=float), color="#2563eb", lw=1.8)
    inset.set_title("trajetoria do objetivo GONM", fontsize=10)
    inset.set_xlabel("iteracao", fontsize=9)
    inset.set_ylabel("energia", fontsize=9)
    inset.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle("GONM | Financas quantitativas e portfolio esparso de alta dimensao", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


def save_outputs(payload: dict, fig: plt.Figure) -> tuple[Path, Path, Path]:
    out_dir = ROOT / "publications" / "T07_GONM" / "results" / "gonm_quant_portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    image_path = out_dir / "gonm_quant_portfolio.png"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# GONM | Quantitative Portfolio Optimization",
                "",
                f"- assets: `{payload['asset_count']}`",
                f"- sectors: `{payload['sector_count']}`",
                f"- baseline objective: `{payload['baseline']['objective']:.6f}`",
                f"- GONM objective: `{payload['gonm']['objective']:.6f}`",
                f"- baseline effective assets >=2%: `{payload['baseline']['effective_assets_2pct']}`",
                f"- GONM effective assets >=2%: `{payload['gonm']['effective_assets_2pct']}`",
                "",
                "## Files",
                "",
                "- image: `publications/T07_GONM/results/gonm_quant_portfolio/gonm_quant_portfolio.png`",
                "- summary: `publications/T07_GONM/results/gonm_quant_portfolio/summary.json`",
            ]
        ),
        encoding="utf-8",
    )
    fig.savefig(image_path, dpi=170)
    return json_path, md_path, image_path


def main() -> None:
    market = build_market()
    optimizer = make_optimizer(market)
    gonm_run = optimizer.optimize(seed=SEED)
    baseline_weights = projected_gradient_baseline(market)
    frontier = random_frontier(market, np.random.default_rng(SEED + 1))

    baseline_summary = summarize_portfolio(market, baseline_weights)
    gonm_summary = summarize_portfolio(market, np.asarray(gonm_run.final_best_positions, dtype=float))
    payload = {
        "seed": SEED,
        "asset_count": market.dimensions,
        "sector_count": len(market.sector_names),
        "asset_names": market.asset_names,
        "sector_names": market.sector_names,
        "baseline": baseline_summary,
        "gonm": gonm_summary,
        "trace_true_energy": [float(v) for v in gonm_run.trace_true_energy],
    }

    fig = render_result(market, frontier, baseline_summary, gonm_summary, gonm_run.trace_true_energy)
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
