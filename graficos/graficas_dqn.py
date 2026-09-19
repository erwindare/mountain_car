"""Generate the DQN learning curve for MountainCar-v0."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mountain_car.agents import DQNAgent


DEFAULT_OUTPUT = Path(__file__).with_name("dqn_curva_aprendizaje.png")


def plot_learning_curve(
    rewards: list[float],
    output: Path,
    moving_average_window: int,
) -> None:
    """Plot episode rewards and a moving average, then save the figure."""
    episodes = np.arange(1, len(rewards) + 1)
    rewards_array = np.asarray(rewards, dtype=float)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(
        episodes,
        rewards_array,
        alpha=0.25,
        linewidth=0.8,
        label="Recompensa por episodio",
    )

    window = min(moving_average_window, len(rewards_array))
    if window > 1:
        average = np.convolve(
            rewards_array,
            np.ones(window) / window,
            mode="valid",
        )
        average_episodes = np.arange(window, len(rewards_array) + 1)
        plt.plot(
            average_episodes,
            average,
            linewidth=2,
            label=f"Media movil ({window} episodios)",
        )

    plt.axhline(-200, color="tab:red", linestyle="--", linewidth=1, label="Limite: -200")
    plt.title("Curva de aprendizaje de DQN en MountainCar-v0")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=2_500,
        help="Numero de episodios de entrenamiento (por defecto: 2500).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Ventana de la media movil (por defecto: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Archivo PNG de salida.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes debe ser mayor que cero")
    if args.window < 1:
        parser.error("--window debe ser mayor que cero")

    agent = DQNAgent("MountainCar-v0")
    rewards = agent.train(total_episodes=args.episodes)
    plot_learning_curve(rewards, args.output, args.window)
    print(f"Grafica guardada en: {args.output}")


if __name__ == "__main__":
    main()