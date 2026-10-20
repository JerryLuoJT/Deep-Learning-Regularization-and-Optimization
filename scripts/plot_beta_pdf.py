"""Plot the Beta(0.2, 0.2) probability density function (Section 6, 5%)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import numpy as np

ALPHA = 0.2


def main():
    x = np.linspace(0.001, 0.999, 1000)
    const = math.gamma(2 * ALPHA) / math.gamma(ALPHA) ** 2
    pdf = const * x ** (ALPHA - 1) * (1 - x) ** (ALPHA - 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, pdf, color="tab:blue", label=f"Beta({ALPHA}, {ALPHA})")
    ax.fill_between(x, pdf, alpha=0.2, color="tab:blue")
    ax.set_xlabel("lambda (mixup interpolation coefficient)")
    ax.set_ylabel("probability density")
    ax.set_title(f"Probability density of Beta({ALPHA}, {ALPHA}) used by mixup")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "runs" / "beta_pdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
