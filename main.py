# main.py

import json
import numpy as np
from models.van_laar import VanLaarModel
from models.wilson import WilsonModel

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

    return converted




def choose_model():
    print("Choose activity coefficient model:")
    print("1. Van Laar")
    print("2. Wilson")
    print("3. NRTL (coming soon)")
    print("4. UNIQUAC (coming soon)")
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

    # Only Van Laar implemented for now
    if choice == "1":
        model = VanLaarModel(params["van_laar"])
    if choice == "2":
        model = WilsonModel(params["wilson"])
    else:
        print("Model not implemented yet.")
        return

    # Get any number of components
    x = get_mole_fractions()

    # Compute activity coefficients
    gamma = model.gamma(x)

    print("\nActivity coefficients:")
    for i, g in enumerate(gamma, start=1):
        print(f"γ{i} = {g:.4f}")


if __name__ == "__main__":
    main()
