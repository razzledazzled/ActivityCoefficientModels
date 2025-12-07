import json
import numpy as np
from models.van_laar import VanLaarModel
from models.wilson import WilsonModel
from models.nrtl import NRTLModel
from models.uniquac import UNIQUACModel
from utils.plot import generate_compositions, compute_gammas, save_to_csv, plot_ternary_surface, plot_activity_surface, plot_combined_ternary

def load_params():
    """
    Load parameters from JSON for all models.
    Convert 'i,j' keys to tuple(int, int).
    """
    with open("data/parameters.json", "r") as f:
        raw = json.load(f)

    converted = {}

    # Van Laar
    if "van_laar" in raw:
        A = {}
        for k, v in raw["van_laar"]["A"].items():
            i, j = map(int, k.split(","))
            A[(i, j)] = v
        converted["van_laar"] = {"A": A}

    # Wilson
    if "wilson" in raw:
        Lambda = {}
        for k, v in raw["wilson"]["Lambda"].items():
            i, j = map(int, k.split(","))
            Lambda[(i, j)] = v
        converted["wilson"] = {"Lambda": Lambda}

    # --- NRTL ---
    if "nrtl" in raw:
        tau = {}
        alpha = {}
        for k, v in raw["nrtl"]["tau"].items():
            i, j = map(int, k.split(","))
            tau[(i, j)] = v
        for k, v in raw["nrtl"]["alpha"].items():
            i, j = map(int, k.split(","))
            alpha[(i, j)] = v
        converted["nrtl"] = {"tau": tau, "alpha": alpha}

    # --- UNIQUAC ---
    if "uniquac" in raw:
        r = {int(k): v for k, v in raw["uniquac"]["r"].items()}
        q = {int(k): v for k, v in raw["uniquac"]["q"].items()}
        a = {}
        for k, v in raw["uniquac"]["a"].items():
            i, j = map(int, k.split(","))
            a[(i, j)] = v
        converted["uniquac"] = {"r": r, "q": q, "a": a}

    return converted


def choose_model():
    print("Choose activity coefficient model:")
    print("1. Van Laar")
    print("2. Wilson")
    print("3. NRTL")
    print("4. UNIQUAC")
    print("5. All Models (Coming Soon)")
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
    x /= total  # normalize automatically

    return x.tolist()


def main():
    params = load_params()
    choice = choose_model()

    # Select model
    if choice == "1":
        model = VanLaarModel(params["van_laar"])
        model_name = "van_laar"
    elif choice == "2":
        model = WilsonModel(params["wilson"])
        model_name = "wilson"
    elif choice == "3":
        model = NRTLModel(params["nrtl"])
        model_name = "nrtl"
    elif choice == "4":
        model = UNIQUACModel(params["uniquac"])
        model_name = "uniquac"

    else:
        print("Model not implemented yet.")
        return

    # Get mole fractions from user
    x = get_mole_fractions()

    # Compute activity coefficients for user input
    gamma = model.gamma(x)
    print("\nActivity coefficients:")
    for i, g in enumerate(gamma, start=1):
        print(f"γ{i} = {g:.4f}")

    # Generate a grid of compositions
    compositions = generate_compositions(len(x), points=30)

    # Compute gamma values for grid
    gammas = compute_gammas(model, compositions)

    # Save results to dynamic CSV file
    csv_filename = f"{model_name}_output.csv"
    save_to_csv(compositions, gammas, filename=csv_filename)

    # # Plot ternary surfaces if mixture is ternary
    # if len(x) == 3:
    #     for i in range(3):
    #         plot_ternary_surface(compositions, gammas, component_index=i)

    # # Works for binary and ternary automatically
    # for i in range(len(x)):
    #     plot_activity_surface(compositions, gammas, component_index=i)


    if len(x) == 3:
        plot_combined_ternary(compositions, gammas)
    else:
        for i in range(len(x)):
            plot_activity_surface(compositions, gammas, component_index=i)




if __name__ == "__main__":
    main()
