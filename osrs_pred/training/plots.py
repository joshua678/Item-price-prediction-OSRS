from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_history(history: Dict[str, List[float]], outpath: Path) -> None:
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Pinball loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history["val_cov_err_mean_H"], label="H")
    plt.plot(history["val_cov_err_mean_L"], label="L")
    plt.title("Val mean abs coverage error")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history["val_width_10_90_H"], label="H")
    plt.plot(history["val_width_10_90_L"], label="L")
    plt.title("Val mean width (0.9 - 0.1)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
