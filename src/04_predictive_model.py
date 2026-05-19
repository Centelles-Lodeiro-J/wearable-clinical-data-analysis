import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main() -> None:
    input_path = Path("data/processed/cleaned_wearable_clinical_data.csv")
    df = pd.read_csv(input_path)

    outcome = "deterioration_14d"

    features = [
        "age",
        "sex",
        "comorbidity_count",
        "prior_hospitalisation",
        "mean_heart_rate",
        "sdnn",
        "rmssd",
        "cough_count",
        "irregular_rhythm_burden",
        "ecg_quality_score",
        "valid_ecg_recordings",
    ]

    X = df[features]
    y = df[outcome]

    categorical_features = ["sex"]
    numeric_features = [feature for feature in features if feature not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predicted_probability = model.predict_proba(X_test)[:, 1]
    predicted_class = model.predict(X_test)

    auc = roc_auc_score(y_test, predicted_probability)
    accuracy = accuracy_score(y_test, predicted_class)
    matrix = confusion_matrix(y_test, predicted_class)

    print("Model performance")
    print("-----------------")
    print(f"AUC: {auc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(matrix)

    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        {
            "metric": ["AUC", "Accuracy"],
            "value": [auc, accuracy],
        }
    )

    results.to_csv(output_dir / "model_performance.csv", index=False)


if __name__ == "__main__":
    main()