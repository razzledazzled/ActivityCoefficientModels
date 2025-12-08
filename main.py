# main.py

import json
import numpy as np
from models.van_laar import VanLaarModel
from models.wilson import WilsonModel
from models.nrtl import NRTLModel
from models.uniquac import UNIQUACModel
from utils.plot import (
    generate_compositions,
    compute_gammas,
    save_to_csv,
    plot_activity_surface,
    plot_combined_binary,
    plot_combined_ternary
)


# ---------- Load parameters ----------
def load_params():
    with open("data/parameters.json", "r") as f:
        raw = json.load(f)

    converted = {}

    # Van Laar
    if "Van_Laar" in raw:
        A = {(int(i), int(j)): v for (i, j), v in 
             ((k.split(","), v) for k, v in raw["Van_Laar"]["A"].items())}
        converted["Van_Laar"] = {"A": A}

    # Wilson
    if "Wilson" in raw:
        Lambda = {(int(i), int(j)): v for (i, j), v in 
                  ((k.split(","), v) for k, v in raw["Wilson"]["Lambda"].items())}
        converted["Wilson"] = {"Lambda": Lambda}

    # NRTL
    if "NRTL" in raw:
        tau = {(int(i), int(j)): v for (i, j), v in 
               ((k.split(","), v) for k, v in raw["NRTL"]["tau"].items())}
        alpha = {(int(i), int(j)): v for (i, j), v in 
                 ((k.split(","), v) for k, v in raw["NRTL"]["alpha"].items())}
        converted["NRTL"] = {"tau": tau, "alpha": alpha}

    # UNIQUAC
    if "UNIQUAC" in raw:
        r = {int(k): v for k, v in raw["UNIQUAC"]["r"].items()}
        q = {int(k): v for k, v in raw["UNIQUAC"]["q"].items()}
        a = {(int(i), int(j)): v for (i, j), v in 
             ((k.split(","), v) for k, v in raw["UNIQUAC"]["a"].items())}
        converted["UNIQUAC"] = {"r": r, "q": q, "a": a}

    return converted


# ---------- Menus ----------
def choose_model():
    print("Choose activity coefficient model:")
    print("1. Van Laar")
    print("2. Wilson")
    print("3. NRTL")
    print("4. UNIQUAC")
    print("5. All Models")
    return input("Enter number: ")


def get_mole_fractions():
    N = int(input("\nEnter number of components: "))
    x = []

    for i in range(N):
        xi = float(input(f"x{i+1}: "))
        x.append(xi)

    x = np.array(x)
    total = x.sum()

    if total <= 0:
        raise ValueError("Sum of mole fractions must be positive")

    x /= total
    return x.tolist()


# ---------- Main ----------
def main():
    params = load_params()
    choice = choose_model()
    x = get_mole_fractions()

    # Map choices to model constructors
    model_map = {
        "1": ("Van_Laar", VanLaarModel),
        "2": ("Wilson", WilsonModel),
        "3": ("NRTL", NRTLModel),
        "4": ("UNIQUAC", UNIQUACModel),
    }

    # Build list of models to run
    if choice in model_map:
        name, cls = model_map[choice]
        models_to_run = [(name, cls(params[name]))]
    elif choice == "5":
        models_to_run = [
            ("Van_Laar", VanLaarModel(params["Van_Laar"])),
            ("Wilson", WilsonModel(params["Wilson"])),
            ("NRTL", NRTLModel(params["NRTL"])),
            ("UNIQUAC", UNIQUACModel(params["UNIQUAC"])),
        ]
    else:
        print("Invalid choice.")
        return

    # Run each model
    for model_name, model in models_to_run:

        print(f"\n--- {model_name.upper()} MODEL ---")

        # Single-point calculation
        gamma = model.gamma(x)
        print("Activity coefficients:")
        for i, g in enumerate(gamma, 1):
            print(f"γ{i} = {g:.4f}")

        # Grid calculations
        compositions = generate_compositions(len(x), points=30)
        gammas = compute_gammas(model, compositions)

        # Save CSV
        csv_filename = f"{model_name}_output.csv"
        save_to_csv(compositions, gammas, filename=csv_filename)

        # Plotting
        if len(x) == 3:
        # Ternary → one combined 3D plot
            plot_combined_ternary(compositions, gammas, model_name=model_name)

        elif len(x) == 2:
        # Binary → one combined 2D plot
            plot_combined_binary(compositions, gammas, model_name=model_name)

        else:
        # Fallback for >3 components
            for i in range(len(x)):
                plot_activity_surface(
                    compositions, gammas,
                    model_name=model_name,
                    component_index=i
                )



if __name__ == "__main__":
    main()
