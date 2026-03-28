import json
import sys
from dataclasses import dataclass
from hashlib import sha256
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


SEED = 131
RESULT_DIR = ROOT / "publications" / "T07_GONM" / "results" / "gonm_chaotic_crypto"
FIGURE_PATH = RESULT_DIR / "gonm_chaotic_crypto.png"
SUMMARY_JSON_PATH = RESULT_DIR / "summary.json"
SUMMARY_MD_PATH = RESULT_DIR / "summary.md"


@dataclass(slots=True)
class ContractiveChaoticChannel:
    coupling: float = 0.64
    drive_mix: float = 0.88
    noise_std: float = 0.006

    def chaotic_map(self, x: float) -> float:
        return 4.0 * x * (1.0 - x)

    def sender_step(self, x: float, message_bit: int, rng: np.random.Generator) -> tuple[float, float]:
        chaotic = self.chaotic_map(x)
        symbol = chaotic + 0.13 * (2 * message_bit - 1) + rng.normal(scale=self.noise_std)
        return float(np.clip(chaotic, 1e-6, 1.0 - 1e-6)), float(symbol)

    def receiver_step(self, y: float, observed_symbol: float) -> float:
        local = self.chaotic_map(y)
        update = self.coupling * local + (1.0 - self.coupling) * observed_symbol + self.drive_mix * (observed_symbol - local)
        return float(np.clip(update, 1e-6, 1.0 - 1e-6))

    def attacker_step(self, z: float, observed_symbol: float, rng: np.random.Generator) -> float:
        # Attacker receives the same public symbol but lacks the stabilizing contractive update.
        local = self.chaotic_map(z)
        guessed = 0.82 * local + 0.18 * (observed_symbol + rng.normal(scale=0.07))
        return float(np.clip(guessed, 1e-6, 1.0 - 1e-6))


def text_to_bits(text: str) -> list[int]:
    payload = text.encode("utf-8")
    bits: list[int] = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        if len(chunk) < 8:
            break
        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)
        out.append(value)
    return bytes(out)


def stable_key_from_state(x: float) -> str:
    raw = f"{x:.16f}".encode("utf-8")
    return sha256(raw).hexdigest()[:32]


def safe_text(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:-_!?()/[]{}")
    return "".join(ch if ch in allowed else "." for ch in text)


def run_demo(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    channel = ContractiveChaoticChannel()
    message = "CPP stabilizes lawful decoding."
    message_bits = text_to_bits(message)

    x = 0.217391
    y = 0.843211
    z = 0.491337

    sender_states = [x]
    receiver_states = [y]
    attacker_states = [z]
    observed = []
    bit_errors_receiver = []
    bit_errors_attacker = []
    receiver_bits: list[int] = []
    attacker_bits: list[int] = []

    for bit in message_bits:
        x, symbol = channel.sender_step(x, bit, rng)
        y = channel.receiver_step(y, symbol)
        z = channel.attacker_step(z, symbol, rng)

        decoded_receiver = int(symbol > channel.chaotic_map(y))
        decoded_attacker = int(symbol > channel.chaotic_map(z))

        receiver_bits.append(decoded_receiver)
        attacker_bits.append(decoded_attacker)
        bit_errors_receiver.append(int(decoded_receiver != bit))
        bit_errors_attacker.append(int(decoded_attacker != bit))

        sender_states.append(x)
        receiver_states.append(y)
        attacker_states.append(z)
        observed.append(symbol)

    sender_arr = np.asarray(sender_states, dtype=float)
    receiver_arr = np.asarray(receiver_states, dtype=float)
    attacker_arr = np.asarray(attacker_states, dtype=float)
    sync_error_receiver = np.abs(sender_arr - receiver_arr)
    sync_error_attacker = np.abs(sender_arr - attacker_arr)

    receiver_text = safe_text(bits_to_bytes(receiver_bits).decode("utf-8", errors="replace"))
    attacker_text = safe_text(bits_to_bytes(attacker_bits).decode("utf-8", errors="replace"))

    return {
        "message": message,
        "message_bits": len(message_bits),
        "sender_states": sender_arr.tolist(),
        "receiver_states": receiver_arr.tolist(),
        "attacker_states": attacker_arr.tolist(),
        "observed_symbols": observed,
        "receiver_bit_errors": int(sum(bit_errors_receiver)),
        "attacker_bit_errors": int(sum(bit_errors_attacker)),
        "receiver_ber": float(np.mean(bit_errors_receiver)),
        "attacker_ber": float(np.mean(bit_errors_attacker)),
        "receiver_text": receiver_text,
        "attacker_text": attacker_text,
        "final_sender_key": stable_key_from_state(sender_arr[-1]),
        "final_receiver_key": stable_key_from_state(receiver_arr[-1]),
        "final_attacker_key": stable_key_from_state(attacker_arr[-1]),
        "sync_error_receiver_final": float(sync_error_receiver[-1]),
        "sync_error_attacker_final": float(sync_error_attacker[-1]),
        "sync_error_receiver_mean": float(np.mean(sync_error_receiver)),
        "sync_error_attacker_mean": float(np.mean(sync_error_attacker)),
    }


def render_result(result: dict) -> plt.Figure:
    sender = np.asarray(result["sender_states"], dtype=float)
    receiver = np.asarray(result["receiver_states"], dtype=float)
    attacker = np.asarray(result["attacker_states"], dtype=float)
    t_state = np.arange(len(sender))
    t_symbol = np.arange(len(result["observed_symbols"]))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.1])
    ax_states = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[0, 1])
    ax_symbols = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    ax_states.plot(t_state, sender, color="#111827", linewidth=2.0, label="sender")
    ax_states.plot(t_state, receiver, color="#2563eb", linewidth=2.0, label="authorized receiver")
    ax_states.plot(t_state, attacker, color="#ef4444", linewidth=1.8, label="attacker")
    ax_states.set_title("Sincronizacao de estados caoticos")
    ax_states.set_xlabel("passo")
    ax_states.set_ylabel("estado interno")
    ax_states.grid(True, linestyle=":", alpha=0.45)
    ax_states.legend(loc="best")

    ax_error.plot(t_state, np.abs(sender - receiver), color="#2563eb", linewidth=2.0, label="receiver sync error")
    ax_error.plot(t_state, np.abs(sender - attacker), color="#ef4444", linewidth=2.0, label="attacker sync error")
    ax_error.set_yscale("log")
    ax_error.set_title("Erro de sincronizacao")
    ax_error.set_xlabel("passo")
    ax_error.set_ylabel("|x - x_hat|")
    ax_error.grid(True, linestyle=":", alpha=0.45)
    ax_error.legend(loc="best")

    ax_symbols.plot(t_symbol, result["observed_symbols"], color="#64748b", linewidth=1.6, label="public chaotic symbol")
    ax_symbols.set_title("Canal observado")
    ax_symbols.set_xlabel("passo")
    ax_symbols.set_ylabel("simbolo publico")
    ax_symbols.grid(True, linestyle=":", alpha=0.45)
    ax_symbols.legend(loc="best")

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        "\n".join(
            [
                "GONM/CPP | chaotic contractive channel",
                "",
                f"message bits = {result['message_bits']}",
                f"receiver BER = {result['receiver_ber']:.4f}",
                f"attacker BER = {result['attacker_ber']:.4f}",
                "",
                f"receiver final sync error = {result['sync_error_receiver_final']:.6e}",
                f"attacker final sync error = {result['sync_error_attacker_final']:.6e}",
                "",
                f"sender key = {result['final_sender_key']}",
                f"receiver key = {result['final_receiver_key']}",
                f"attacker key = {result['final_attacker_key']}",
                "",
                "receiver text:",
                result["receiver_text"],
                "",
                "attacker text:",
                result["attacker_text"],
                "",
                "Observacao:",
                "esta e uma demo de sincronizacao contrativa,",
                "nao um esquema criptografico real para producao.",
            ]
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        family="monospace",
    )

    fig.suptitle("GONM | Contractive Chaotic Synchronization Demo", fontsize=16, y=0.98)
    fig.tight_layout()
    return fig


def write_summary(result: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = f"""# GONM | Chaotic Contractive Synchronization

This simulation is an educational demonstration of contractive synchronization over a chaotic channel.

## Recorded outcome

- receiver BER: `{result["receiver_ber"]:.6f}`
- attacker BER: `{result["attacker_ber"]:.6f}`
- receiver final synchronization error: `{result["sync_error_receiver_final"]:.6e}`
- attacker final synchronization error: `{result["sync_error_attacker_final"]:.6e}`

## Interpretation

This is not a production cryptographic primitive and should not be used to secure real communications. The narrower point is conceptual: a contractive update law can let an authorized receiver stabilize onto the sender's internal dynamics while an outsider remains desynchronized.
"""
    SUMMARY_MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    result = run_demo(seed=SEED)
    fig = render_result(result)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    if "--no-show" not in sys.argv and "--save" not in sys.argv:
        plt.show()
    plt.close(fig)
    write_summary(result)


if __name__ == "__main__":
    main()
