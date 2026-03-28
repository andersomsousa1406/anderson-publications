from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan, cos, erf, exp, log, pi, sin, sqrt
from pathlib import Path
from time import perf_counter
import json

import numpy as np


@dataclass
class Scenario:
    name: str
    category: str
    a: float
    b: float
    positive: bool
    discontinuous: bool
    func: object
    exact: object


@dataclass
class MethodStat:
    scenario: str
    category: str
    method: str
    n: int
    estimate: float | None
    abs_error: float | None
    runtime_ms: float
    evals: int
    valid: int


@dataclass
class ConvergenceRow:
    scenario: str
    method: str
    order_estimate: float | None
    first_error: float | None
    last_error: float | None
    monotone_improvement: int


@dataclass
class NoiseRow:
    scenario: str
    method: str
    n: int
    mean_abs_error: float | None
    std_abs_error: float | None
    valid_rate: float


class EvalCounter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, x):
        arr = np.asarray(x)
        self.count += int(arr.size)
        return self.func(arr)


def exact_integral_step(a: float, b: float, threshold: float = 0.3) -> float:
    left = max(a, threshold)
    return max(0.0, b - left)


def exact_integral_sign(a: float, b: float) -> float:
    left = max(a, 0.0) - a
    right = b - max(a, 0.0)
    neg = min(max(0.0, min(b, 0.0) - a), b - a)
    pos = max(0.0, b - max(a, 0.0))
    return pos - neg


def exact_integral_sin(a: float, b: float, k: float = 1.0) -> float:
    return (-cos(k * b) + cos(k * a)) / k


def exact_integral_cos(a: float, b: float, k: float = 1.0) -> float:
    return (sin(k * b) - sin(k * a)) / k


def exact_integral_exp(a: float, b: float, c: float = 1.0) -> float:
    return (exp(c * b) - exp(c * a)) / c


def exact_integral_poly2(a: float, b: float) -> float:
    return (b**3 - a**3) / 3.0


def exact_integral_poly3(a: float, b: float) -> float:
    return (b**4 - a**4) / 4.0


def exact_integral_sqrt(a: float, b: float) -> float:
    return (2.0 / 3.0) * (b ** 1.5 - a ** 1.5)


def exact_integral_log(a: float, b: float) -> float:
    return (b * log(b) - b) - (a * log(a) - a)


def exact_integral_rational_peak(a: float, b: float) -> float:
    return (atan(10.0 * b) - atan(10.0 * a)) / 10.0


def exact_integral_sign_change_poly(a: float, b: float) -> float:
    return (b**4 - a**4) / 4.0 - (b**2 - a**2) / 2.0


def exact_integral_sin_cos(a: float, b: float) -> float:
    return 0.25 * (cos(2.0 * a) - cos(2.0 * b))


def exact_velocity_distance_1(a: float, b: float) -> float:
    return 2.0 * (b - a) + (-cos(3.0 * b) + cos(3.0 * a)) / 3.0 + 0.1 * (b**2 - a**2)


def exact_velocity_distance_2(a: float, b: float) -> float:
    gauss = 0.25 * sqrt(pi) * (erf(b - 5.0) - erf(a - 5.0))
    osc = (-0.3 / 8.0) * (cos(8.0 * b) - cos(8.0 * a))
    return (b - a) + gauss + osc


def reference_gauss(func, a: float, b: float, order: int = 256) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    return 0.5 * (b - a) * float(np.dot(weights, func(x)))


def logarithmic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a, dtype=float)
    close = np.isclose(a, b, rtol=1e-12, atol=1e-15)
    out[close] = a[close]
    mask = ~close
    out[mask] = (b[mask] - a[mask]) / (np.log(b[mask]) - np.log(a[mask]))
    return out


def composite_midpoint(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = a + (np.arange(n, dtype=float) + 0.5) * h
    return h * float(np.sum(f(x)))


def composite_trapezoid(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * float(0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])


def composite_simpson(f, a: float, b: float, n: int) -> float:
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h / 3.0 * float(y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2]))


def composite_geometric(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    if np.any(y <= 0):
        raise ValueError('nonpositive values')
    return h * float(np.sum(np.sqrt(y[:-1] * y[1:])))


def composite_mqlm(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    if np.any(y <= 0):
        raise ValueError('nonpositive values')
    lm = logarithmic_mean(y[:-1], y[1:])
    return h * float(np.sum(lm))


def composite_mqlm_richardson(f, a: float, b: float, n: int) -> float:
    return (4.0 * composite_mqlm(f, a, b, 2 * n) - composite_mqlm(f, a, b, n)) / 3.0


def composite_weighted_blend(f, a: float, b: float, n: int, weight: float = 0.5) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    if np.any(y <= 0):
        raise ValueError('nonpositive values')
    arithmetic = 0.5 * (y[:-1] + y[1:])
    lm = logarithmic_mean(y[:-1], y[1:])
    blend = weight * lm + (1.0 - weight) * arithmetic
    return h * float(np.sum(blend))


def composite_adaptive_blend(f, a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    y = f(x)
    ym = f(xm)
    if np.any(y <= 0) or np.any(ym <= 0):
        return composite_trapezoid(f, a, b, n)
    arithmetic = 0.5 * (y[:-1] + y[1:])
    lm = logarithmic_mean(y[:-1], y[1:])
    log_mid = 0.5 * (np.log(y[:-1]) + np.log(y[1:]))
    mismatch = np.abs(np.log(ym) - log_mid)
    scale = np.maximum(np.max(mismatch), 1e-12)
    w = np.exp(-4.0 * mismatch / scale)
    blend = w * lm + (1.0 - w) * arithmetic
    return h * float(np.sum(blend))


def gauss_legendre_32(f, a: float, b: float, _n: int) -> float:
    return reference_gauss(f, a, b, order=32)


def get_methods():
    return {
        'midpoint': composite_midpoint,
        'trapezoid': composite_trapezoid,
        'simpson': composite_simpson,
        'geometric': composite_geometric,
        'mqlm': composite_mqlm,
        'mqlm_rich': composite_mqlm_richardson,
        'weighted_blend': composite_weighted_blend,
        'adaptive_blend': composite_adaptive_blend,
        'gauss32': gauss_legendre_32,
    }


def build_scenarios() -> list[Scenario]:
    return [
        Scenario('sin_0_pi', 'smooth', 0.0, pi, False, False, lambda x: np.sin(x), lambda a, b: exact_integral_sin(a, b, 1.0)),
        Scenario('exp_0_1', 'smooth', 0.0, 1.0, True, False, lambda x: np.exp(x), lambda a, b: exact_integral_exp(a, b, 1.0)),
        Scenario('x2_0_1', 'smooth', 0.0, 1.0, True, False, lambda x: x * x, exact_integral_poly2),
        Scenario('x3_m1_1', 'smooth', -1.0, 1.0, False, False, lambda x: x**3, exact_integral_poly3),
        Scenario('sqrt_0_1', 'singularity', 0.0, 1.0, True, False, lambda x: np.sqrt(x), exact_integral_sqrt),
        Scenario('log_eps_1', 'singularity', 1.0e-6, 1.0, False, False, lambda x: np.log(x), exact_integral_log),
        Scenario('exp10_0_1', 'high_curvature', 0.0, 1.0, True, False, lambda x: np.exp(10.0 * x), lambda a, b: exact_integral_exp(a, b, 10.0)),
        Scenario('rational_peak', 'high_curvature', -1.0, 1.0, True, False, lambda x: 1.0 / (1.0 + 100.0 * x * x), exact_integral_rational_peak),
        Scenario('sin10_0_1', 'oscillatory', 0.0, 1.0, False, False, lambda x: np.sin(10.0 * x), lambda a, b: exact_integral_sin(a, b, 10.0)),
        Scenario('sin50_0_1', 'oscillatory', 0.0, 1.0, False, False, lambda x: np.sin(50.0 * x), lambda a, b: exact_integral_sin(a, b, 50.0)),
        Scenario('cos100_0_1', 'oscillatory', 0.0, 1.0, False, False, lambda x: np.cos(100.0 * x), lambda a, b: exact_integral_cos(a, b, 100.0)),
        Scenario('x3_minus_x', 'sign_change', -1.0, 1.0, False, False, lambda x: x**3 - x, exact_integral_sign_change_poly),
        Scenario('sin_cos', 'sign_change', 0.0, pi, False, False, lambda x: np.sin(x) * np.cos(x), exact_integral_sin_cos),
        Scenario('step_0_1', 'discontinuous', 0.0, 1.0, False, True, lambda x: (x >= 0.3).astype(float), lambda a, b: exact_integral_step(a, b, 0.3)),
        Scenario('sign_m1_1', 'discontinuous', -1.0, 1.0, False, True, lambda x: np.where(x > 0.0, 1.0, np.where(x < 0.0, -1.0, 0.0)), exact_integral_sign),
        Scenario('velocity_distance_1', 'simulated_real', 0.0, 10.0, True, False, lambda x: 2.0 + np.sin(3.0 * x) + 0.2 * x, exact_velocity_distance_1),
        Scenario('velocity_distance_2', 'simulated_real', 0.0, 10.0, True, False, lambda x: 1.0 + 0.5 * np.exp(-(x - 5.0) ** 2) + 0.3 * np.sin(8.0 * x), exact_velocity_distance_2),
    ]


def reference_value(scenario: Scenario) -> float:
    if scenario.exact is not None:
        return float(scenario.exact(scenario.a, scenario.b))
    return reference_gauss(scenario.func, scenario.a, scenario.b, order=256)


def run_method(method_name: str, method, scenario: Scenario, n: int, noisy: bool = False, seed: int = 0) -> MethodStat:
    base_func = scenario.func
    if noisy:
        grid = np.linspace(scenario.a, scenario.b, n + 1)
        y_grid = base_func(grid)
        rng = np.random.default_rng(seed)
        if scenario.positive:
            rel_noise = rng.normal(scale=0.02, size=y_grid.shape)
            y_noisy = np.maximum(y_grid * (1.0 + rel_noise), 1e-12)
        else:
            amp = max(1e-8, 0.02 * max(1.0, float(np.max(np.abs(y_grid)))))
            y_noisy = y_grid + rng.normal(scale=amp, size=y_grid.shape)
        def interp_func(x):
            return np.interp(np.asarray(x, dtype=float), grid, y_noisy)
        func = interp_func
    else:
        func = base_func

    counted = EvalCounter(func)
    start = perf_counter()
    try:
        estimate = float(method(counted, scenario.a, scenario.b, n))
        ref = reference_value(scenario)
        error = abs(estimate - ref)
        valid = 1
    except Exception:
        estimate = None
        error = None
        valid = 0
    runtime_ms = 1000.0 * (perf_counter() - start)
    return MethodStat(scenario.name, scenario.category, method_name, n, estimate, error, runtime_ms, counted.count, valid)


def summarize_convergence(stats: list[MethodStat], scenario_name: str, method_name: str) -> ConvergenceRow:
    rows = [s for s in stats if s.scenario == scenario_name and s.method == method_name and s.valid and s.abs_error is not None]
    rows.sort(key=lambda r: r.n)
    if len(rows) < 2:
        return ConvergenceRow(scenario_name, method_name, None, None, None, 0)
    hs = np.array([1.0 / r.n for r in rows], dtype=float)
    errs = np.array([max(r.abs_error, 1e-30) for r in rows], dtype=float)
    slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
    mono = int(np.all(np.diff(errs) <= 1e-15))
    return ConvergenceRow(scenario_name, method_name, float(slope), float(errs[0]), float(errs[-1]), mono)


def summarize_noise(stats: list[MethodStat], scenario_name: str, method_name: str, n: int) -> NoiseRow:
    rows = [s for s in stats if s.scenario == scenario_name and s.method == method_name and s.n == n]
    valid_rows = [r for r in rows if r.valid and r.abs_error is not None]
    if not valid_rows:
        return NoiseRow(scenario_name, method_name, n, None, None, 0.0)
    errs = np.array([r.abs_error for r in valid_rows], dtype=float)
    return NoiseRow(scenario_name, method_name, n, float(np.mean(errs)), float(np.std(errs)), len(valid_rows) / len(rows))


def build_markdown(stats: list[MethodStat], convergence: list[ConvergenceRow], noise_rows: list[NoiseRow], out_dir: Path) -> None:
    methods = ['midpoint', 'trapezoid', 'simpson', 'geometric', 'mqlm', 'mqlm_rich', 'weighted_blend', 'adaptive_blend', 'gauss32']
    scenarios = build_scenarios()
    fine_n = 100
    lines = ['# Benchmark Amplo de Quadratura', '', 'Metodos comparados: Midpoint, Trapezoid, Simpson, Geometric, MQLM, MQLM+Richardson, Weighted Blend, Adaptive Blend e Gauss32.', '']

    lines += ['## Resumo por categoria no grid medio (n=100)', '']
    categories = sorted({s.category for s in scenarios})
    for category in categories:
        lines.append(f'### `{category}`')
        cat_rows = [r for r in stats if r.category == category and r.n == fine_n and r.valid and r.abs_error is not None]
        for method in methods:
            mrows = [r for r in cat_rows if r.method == method]
            if not mrows:
                lines.append(f'- `{method}`: sem casos validos')
                continue
            err = float(np.mean([r.abs_error for r in mrows]))
            tms = float(np.mean([r.runtime_ms for r in mrows]))
            evals = float(np.mean([r.evals for r in mrows]))
            lines.append(f'- `{method}`: erro medio={err:.3e}, tempo medio={tms:.3f} ms, avaliacoes medias={evals:.1f}')
        lines.append('')

    lines += ['## Convergencia', '']
    for row in convergence:
        if row.order_estimate is not None:
            lines.append(f'- `{row.scenario}` / `{row.method}`: ordem~{row.order_estimate:.2f}, erro_inicial={row.first_error:.3e}, erro_final={row.last_error:.3e}, melhoria_monotona={bool(row.monotone_improvement)}')
    lines.append('')

    lines += ['## Ruido (n=100, media sobre sementes)', '']
    for row in noise_rows:
        if row.mean_abs_error is None:
            lines.append(f'- `{row.scenario}` / `{row.method}`: sem execucoes validas')
        else:
            lines.append(f'- `{row.scenario}` / `{row.method}`: erro_medio={row.mean_abs_error:.3e}, desvio={row.std_abs_error:.3e}, taxa_valida={row.valid_rate:.2%}')
    lines.append('')

    lines += ['## Leitura consolidada', '']
    lines += [
        '- `MQLM` e `MQLM+Richardson` so sao aplicaveis de forma limpa a integrandos positivos.',
        '- em funcoes exponenciais e quase exponenciais, a familia MQLM deve ter vantagem estrutural sobre regras puramente aditivas.',
        '- em funcoes com mudanca de sinal ou descontinuidade, regras multiplicativas deixam de ser a linguagem natural; aqui as comparacoes mais honestas sao contra trapezio e Simpson.',
        '- `weighted_blend` e `adaptive_blend` medem se combinacoes entre media aritmetica e estrutura logaritmica ajudam em regimes mistos.',
        '- `gauss32` entra como comparador classico de alta precisao/custo fixo, nao como metodo em malha uniforme.',
    ]
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    scenarios = build_scenarios()
    methods = get_methods()
    ns = [10, 50, 100, 500, 1000]
    convergence_ns = [10, 20, 40, 80, 160, 320, 640]
    noise_methods = ['trapezoid', 'simpson', 'mqlm', 'mqlm_rich', 'adaptive_blend']

    stats: list[MethodStat] = []
    for scenario in scenarios:
        for n in ns:
            for method_name, method in methods.items():
                stats.append(run_method(method_name, method, scenario, n, noisy=False))

    for scenario in scenarios:
        for n in convergence_ns:
            for method_name in ['trapezoid', 'simpson', 'mqlm', 'mqlm_rich', 'adaptive_blend']:
                stats.append(run_method(method_name, methods[method_name], scenario, n, noisy=False))

    noise_stats: list[MethodStat] = []
    for scenario in scenarios:
        for seed in range(12):
            for method_name in noise_methods:
                noise_stats.append(run_method(method_name, methods[method_name], scenario, 100, noisy=True, seed=seed))

    convergence_rows: list[ConvergenceRow] = []
    for scenario in scenarios:
        for method_name in ['trapezoid', 'simpson', 'mqlm', 'mqlm_rich', 'adaptive_blend']:
            convergence_rows.append(summarize_convergence(stats, scenario.name, method_name))

    noise_rows: list[NoiseRow] = []
    for scenario in scenarios:
        for method_name in noise_methods:
            noise_rows.append(summarize_noise(noise_stats, scenario.name, method_name, 100))

    out_dir = Path(r'c:\Projetos\CSD\out\quadrature_benchmark_results')
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'stats': [asdict(r) for r in stats],
        'noise_stats': [asdict(r) for r in noise_stats],
        'convergence': [asdict(r) for r in convergence_rows],
        'noise_summary': [asdict(r) for r in noise_rows],
    }
    (out_dir / 'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    build_markdown(stats, convergence_rows, noise_rows, out_dir)
    print('Quadrature benchmark suite completed.')
    print(out_dir)


if __name__ == '__main__':
    main()
