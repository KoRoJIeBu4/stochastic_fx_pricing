import matplotlib.pyplot as plt
import numpy as np

from stochastic_fx_pricing.models import Trajectories


def plot_trajectories(
    trajectories: Trajectories,
    max_trajectories: int,
    seed: int,
):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    data = [
        (trajectories.rub_rates * 100, "Мгновенная процентная ставка RUB", "%"),
        (trajectories.usd_rates * 100, "Мгновенная процентная ставка USD", "%"),
        (trajectories.fx_rates, "Обменный курс USD/RUB", "RUB"),
    ]

    rng = np.random.default_rng(seed)

    n_paths = trajectories.fx_rates.shape[0]

    n = min(max_trajectories, n_paths)

    indices = rng.choice(
        n_paths,
        n,
        replace=False,
    )

    for ax, (values, title, unit) in zip(axes, data, strict=False):
        mean = np.mean(values, axis=0)

        ax.plot(
            trajectories.timestamps,
            values[indices].T,
            color="#7796B0",
            alpha=0.12,
            linewidth=0.7,
        )

        ax.plot(
            trajectories.timestamps,
            mean,
            color="#F28C28",
            linewidth=2.5,
            label="Среднее",
            zorder=10,
        )

        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(unit)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()
