# main.py

import json
from models.van_laar import VanLaarModel

def load_params():
    with open("data/parameters.json", "r") as f:
        raw = json.load(f)
    
    # Convert "i,j" keys to tuples ("1","2")
    converted = {}
    A = {}
    for k, v in raw["van_laar"]["A"].items():
        i, j = k.split(",")
        A[(i, j)] = v
    converted["van_laar"] = {"A": A}
    return converted

def choose_model():
    print("Choose activity coefficient model:")
    print("1. Van Laar")
    choice = input("Enter number: ")

    return choice

def main():
    params = load_params()

    choice = choose_model()

    if choice == "1":
        model = VanLaarModel(params["van_laar"])
    else:
        print("Invalid choice")
        return

    print("\nEnter mole fractions for a ternary mixture (must sum to 1):")
    x1 = float(input("x1: "))
    x2 = float(input("x2: "))
    x3 = float(input("x3: "))

    x = [x1, x2, x3]

    gamma = model.gamma(x)

    print("\nActivity coefficients:")
    print(f"γ1 = {gamma[0]:.4f}")
    print(f"γ2 = {gamma[1]:.4f}")
    print(f"γ3 = {gamma[2]:.4f}")

if __name__ == "__main__":
    main()
