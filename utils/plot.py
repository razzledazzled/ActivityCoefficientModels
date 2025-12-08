# utils/plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


# ---------- Composition utilities ----------
def generate_compositions(N, points=30):
    if N == 2:
        x1 = np.linspace(0, 1, points)
        return [np.array([x, 1-x]) for x in x1]
    elif N == 3:
        x1 = np.linspace(0, 1, points)
        x2 = np.linspace(0, 1, points)
        compositions = []
        for a, b in product(x1, x2):
            if a + b <= 1:
                compositions.append(np.array([a, b, 1 - a - b]))
        return compositions
    else:
        fractions = np.linspace(0, 1, points)
        compositions = [np.array(c) for c in product(fractions, repeat=N) if abs(sum(c)-1)<1e-6]
        return compositions


def compute_gammas(model, compositions):
    return [model.gamma(x) for x in compositions]


def save_to_csv(compositions, gammas, filename="output.csv"):
    N = len(compositions[0])
    columns = [f"x{i+1}" for i in range(N)] + [f"gamma{i+1}" for i in range(N)]
    data = np.hstack([np.array(compositions), np.array(gammas)])
    pd.DataFrame(data, columns=columns).to_csv(filename, index=False)
    print(f"Saved results to {filename}")


# ---------- Dynamic plotting helper ----------
def prepare_plot_values(gammas):
    """
    Determine if log scale is needed and normalize for plotting.

    Returns:
        plot_values: np.array with either linear or log10(gamma)
        color_label: str, axis label
        is_log: bool, whether log scale was used
    """
    gam = np.array(gammas)
    gmin, gmax = gam.min(), gam.max()

    # If range is wide (>10×), use log scale
    if gmax / max(gmin, 1e-12) > 10:
        plot_values = np.log10(gam)
        color_label = "log10(γ)"
        is_log = True
    else:
        plot_values = gam
        color_label = "γ"
        is_log = False

    return plot_values, color_label, is_log


# ---------- Plot functions ----------
def plot_activity_surface(compositions, gammas, model_name="", component_index=0):
    comps = np.array(compositions)
    gam = np.array(gammas)[:, component_index]
    plot_values, color_label, _ = prepare_plot_values(gam)

    N = comps.shape[1]

    if N == 2:
        plt.figure(figsize=(7,5))
        plt.plot(comps[:,0], plot_values, marker='o')
        plt.xlabel("x1")
        plt.ylabel(f"{color_label}{component_index+1}")
        plt.title(f"{model_name} – Binary: γ{component_index+1} vs x1")
        plt.grid(True)
        plt.show()

    elif N == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(comps[:,0], comps[:,1], plot_values, c=plot_values, cmap='viridis')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel(f"{color_label}{component_index+1}")
        ax.set_title(f"{model_name} – Ternary surface for γ{component_index+1}")
        fig.colorbar(sc, label=color_label)
        plt.show()
    else:
        print(f"Plotting not implemented for N={N} components.")


def plot_combined_binary(compositions, gammas, model_name=""):
    comps = np.array(compositions)
    gam = np.array(gammas)
    N = comps.shape[1]

    plt.figure(figsize=(8,6))
    for i in range(N):
        plot_values, color_label, _ = prepare_plot_values(gam[:, i])
        plt.plot(comps[:,0], plot_values, label=f"γ{i+1}")
    plt.xlabel("x1")
    plt.ylabel(color_label)
    plt.title(f"{model_name} – Binary Activity Coefficients")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_combined_ternary(compositions, gammas, model_name=""):
    from mpl_toolkits.mplot3d import Axes3D
    comps = np.array(compositions)
    gam = np.array(gammas)
    N = comps.shape[1]

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i in range(N):
        plot_values, color_label, _ = prepare_plot_values(gam[:, i])
        ax.scatter(comps[:,0], comps[:,1], plot_values, color=colors[i % len(colors)],
                   alpha=0.5, label=f'γ{i+1}')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(color_label)
    ax.set_title(f"{model_name} – Combined Ternary Activity Coefficients")
    ax.legend()
    plt.show()
