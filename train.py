import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://github.com/YBIFoundation/Dataset/raw/main/Titanic.csv"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "metrics.txt")
FEATURE_IMPORTANCE_PNG = os.path.join(os.path.dirname(__file__), "..", "feature_importances.png")

def load_data():
    df = pd.read_csv(DATA_URL)
    # Feature engineering
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    # Select features / target
    X = df[['pclass', 'sex', 'age', 'fare', 'family_size', 'is_alone', 'embarked']].copy()
    y = df['survived'].copy()
    return X, y

def build_pipeline():
    numeric_features = ['age', 'fare', 'family_size', 'is_alone', 'pclass']
    categorical_features = ['sex', 'embarked']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[('preprocess', preprocessor),
                          ('model', clf)])
    return pipe

def get_feature_names(preprocessor):
    # Build feature name list after preprocessing
    numeric_features = ['age', 'fare', 'family_size', 'is_alone', 'pclass']
    categorical_features = ['sex', 'embarked']

    num_names = numeric_features
    cat_names = list(preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features))
    return num_names + cat_names

def main():
    print("Loading data ...")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building pipeline ...")
    pipe = build_pipeline()

    print("Training ...")
    pipe.fit(X_train, y_train)

    print("Evaluating ...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "titanic_model.pkl")
    joblib.dump(pipe, model_path)
    print(f"Model saved to: {model_path}")

    # Feature importances (requires mapping through preprocessor)
    try:
        model = pipe.named_steps['model']
        pre = pipe.named_steps['preprocess']
        feature_names = get_feature_names(pre)
        importances = model.feature_importances_
        # Simple plot
        sorted_idx = np.argsort(importances)
        plt.figure(figsize=(8, 6))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PNG, dpi=150)
        print(f"Feature importances saved to: {FEATURE_IMPORTANCE_PNG}")
    except Exception as e:
        print("Could not generate feature importance plot:", e)

if __name__ == "__main__":
    main()
