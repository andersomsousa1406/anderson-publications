from __future__ import annotations

from anderson.presets import lj12_cooling, lj12_fixed, lj38_budget200k, lj38_default


def lj12_fixed_example():
    return lj12_fixed().run()


def lj12_cooling_example():
    return lj12_cooling().run()


def lj38_default_example():
    return lj38_default().run()


def lj38_budget200k_example():
    return lj38_budget200k().run()
