# Clean the data by removing unusual values
import pandas as pd
from pathlib import Path

# Define a function to clean the data based on specific rules to remove implausible heart-rate values, low-quality ECG records, days with too few valid ECG recordings, and non-positive HRV values. This will help ensure that the dataset is more accurate and reliable for subsequent analysis and modeling.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean mock wearable clinical data.

    Cleaning rules:
    - Remove implausible heart-rate values.
    - Remove low-quality ECG records.
    - Remove days with too few valid ECG recordings.
    - Remove non-positive HRV values.
    """

    cleaned = df.copy()

    cleaned = cleaned[
        (cleaned["mean_heart_rate"] >= 35)
        & (cleaned["mean_heart_rate"] <= 220)
    ]

    cleaned = cleaned[cleaned["ecg_quality_score"] >= 0.70]

    cleaned = cleaned[cleaned["valid_ecg_recordings"] >= 3]

    cleaned = cleaned[
        (cleaned["sdnn"] > 0)
        & (cleaned["rmssd"] > 0)
    ]

    return cleaned


def main() -> None:
    input_path = Path("data/raw/mock_wearable_clinical_data.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    cleaned = clean_data(df)

    output_path = output_dir / "cleaned_wearable_clinical_data.csv"
    cleaned.to_csv(output_path, index=False)

    print(f"Original rows: {df.shape[0]}")
    print(f"Cleaned rows: {cleaned.shape[0]}")
    print(f"Rows removed: {df.shape[0] - cleaned.shape[0]}")
    print(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    main()