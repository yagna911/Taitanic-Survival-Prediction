import os
import argparse
import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "titanic_model.pkl")

def main():
    parser = argparse.ArgumentParser(description="Titanic Survival Prediction")

    parser.add_argument("--pclass", type=int, required=True)
    parser.add_argument("--sex", choices=["male", "female"], required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--fare", type=float, required=True)
    parser.add_argument("--family_size", type=int, required=True)
    parser.add_argument("--is_alone", type=int, choices=[0, 1], required=True)
    parser.add_argument("--embarked", choices=["S", "C", "Q"], required=True)

    args = parser.parse_args()

    example = {
        "pclass": args.pclass,
        "sex": args.sex,
        "age": args.age,
        "fare": args.fare,
        "family_size": args.family_size,
        "is_alone": args.is_alone,
        "embarked": args.embarked
    }

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([example])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    print("\n========== RESULT ==========")
    print(f"Prediction : {'Survived' if pred == 1 else 'Did NOT Survive'}")
    print(f"Probability: {proba:.3f}")
    print("============================\n")

if __name__ == "__main__":
    main()
