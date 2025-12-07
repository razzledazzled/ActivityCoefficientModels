# utils/plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


def generate_compositions(N, points=30):
    """
    Generate evenly spaced compositions for N components.
    For N=2, returns linearly spaced fractions.
    For N=3, returns a triangular grid (ternary).
    For N>3, returns a coarse grid using linspace in N dimensions.

    Returns: list of composition arrays
    """
    if N == 2:
        x1 = np.linspace(0, 1, points)
        compositions = [np.array([x, 1-x]) for x in x1]
    elif N == 3:
        x1 = np.linspace(0, 1, points)
        x2 = np.linspace(0, 1, points)
        compositions = []
        for a, b in product(x1, x2):
            if a + b <= 1:
                compositions.append(np.array([a, b, 1 - a - b]))
    else:
        # For N>3, use a coarse grid
        fractions = np.linspace(0, 1, points)
        compositions = []
        for comb in product(fractions, repeat=N):
            if abs(sum(comb) - 1) < 1e-6:
                compositions.append(np.array(comb))
    return compositions


def compute_gammas(model, compositions):
    """
    Compute activity coefficients for a list of compositions.
    
    Returns: list of gamma arrays
    """
    gammas = [model.gamma(x) for x in compositions]
    return gammas


def save_to_csv(compositions, gammas, filename="output.csv"):
    """
    Save compositions and gamma values to CSV
    """
    N = len(compositions[0])
    columns = [f"x{i+1}" for i in range(N)] + [f"gamma{i+1}" for i in range(N)]
    data = np.hstack([np.array(compositions), np.array(gammas)])
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Saved results to {filename}")

def plot_activity_surface(compositions, gammas, component_index=0):
    """
    Plot activity coefficients for binary or ternary mixtures.

    - Binary (N=2): 2D line plot (x1 vs Y)
    - Ternary (N=3): 3D scatter plot (x1, x2, Y)
    """
    comps = np.array(compositions)
    gam = np.array(gammas)[:, component_index]
    N = comps.shape[1]

    if N == 2:
        # Binary mixture → 2D line plot
        plt.figure(figsize=(7,5))
        plt.plot(comps[:,0], gam, marker='o')
        plt.xlabel("x1")
        plt.ylabel(f"γ{component_index+1}")
        plt.title(f"Binary mixture: γ{component_index+1} vs x1")
        plt.grid(True)
        plt.show()

    elif N == 3:
        # Ternary mixture → 3D scatter plot
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(comps[:,0], comps[:,1], gam, c=gam, cmap='viridis')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel(f"γ{component_index+1}")
        ax.set_title(f"Ternary surface for γ{component_index+1}")
        plt.show()

    else:
        print(f"Plotting not implemented for N={N} components. Use slices or projections.")


def plot_combined_ternary(compositions, gammas):
    """
    Plot all three activity coefficient surfaces in ONE 3D plot.
    """
    from mpl_toolkits.mplot3d import Axes3D

    comps = np.array(compositions)
    gam = np.array(gammas)

    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')

    # γ1
    ax.scatter(comps[:,0], comps[:,1], gam[:,0], color='red', alpha=0.5, label='γ1')

    # γ2
    ax.scatter(comps[:,0], comps[:,1], gam[:,1], color='blue', alpha=0.5, label='γ2')

    # γ3
    ax.scatter(comps[:,0], comps[:,1], gam[:,2], color='green', alpha=0.5, label='γ3')

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("γ")
    ax.set_title("Combined Ternary Activity Coefficients")

    ax.legend()
    plt.show()


##Unused

def plot_ternary_surface(compositions, gammas, component_index=0):
    """
    Plot a 3D surface for a ternary mixture.
    component_index: which γ component to plot
    """
    from mpl_toolkits.mplot3d import Axes3D

    comps = np.array(compositions)
    gam = np.array(gammas)[:, component_index]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(comps[:,0], comps[:,1], gam, c=gam, cmap='viridis')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(f"γ{component_index+1}")
    ax.set_title(f"Ternary surface for γ{component_index+1}")
    plt.show()