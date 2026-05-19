import numpy as np
import pandas as pd
from pathlib import Path


def generate_mock_data(n_patients: int = 500, n_days: int = 84, seed: int = 42) -> pd.DataFrame:
    """
    Generate mock patient-day data for a wearable clinical monitoring study.

    Each row represents one patient on one study day.
    The outcome is clinical deterioration within the next 14 days.
    """

    np.random.seed(seed)

    rows = []

    for patient_id in range(1, n_patients + 1):
        age = np.random.normal(63, 12)
        sex = np.random.choice(["Female", "Male"])
        comorbidity_count = np.random.poisson(2)
        prior_hospitalisation = np.random.binomial(1, 0.30)

        patient_risk = (
            0.02 * (age - 60)
            + 0.35 * comorbidity_count
            + 0.80 * prior_hospitalisation
        )

        for day in range(1, n_days + 1):
            mean_heart_rate = np.random.normal(76 + patient_risk * 3, 9)
            sdnn = np.random.normal(42 - patient_risk * 4, 8)
            rmssd = np.random.normal(35 - patient_risk * 3, 7)
            cough_count = np.random.poisson(max(1, 4 + patient_risk))

            irregular_rhythm_burden = np.random.beta(
                1.2 + patient_risk * 0.2,
                15
            )

            ecg_quality_score = np.random.normal(0.88, 0.10)
            ecg_quality_score = np.clip(ecg_quality_score, 0, 1)

            valid_ecg_recordings = np.random.poisson(6 * ecg_quality_score)

            log_odds = (
                -4.0
                + 0.035 * (mean_heart_rate - 75)
                - 0.045 * (sdnn - 40)
                + 0.90 * irregular_rhythm_burden
                + 0.04 * cough_count
                + 0.30 * comorbidity_count
                + 0.60 * prior_hospitalisation
            )

            probability = 1 / (1 + np.exp(-log_odds))
            deterioration_14d = np.random.binomial(1, probability)

            rows.append(
                {
                    "patient_id": patient_id,
                    "study_day": day,
                    "age": round(age, 1),
                    "sex": sex,
                    "comorbidity_count": comorbidity_count,
                    "prior_hospitalisation": prior_hospitalisation,
                    "mean_heart_rate": round(mean_heart_rate, 1),
                    "sdnn": round(sdnn, 1),
                    "rmssd": round(rmssd, 1),
                    "cough_count": cough_count,
                    "irregular_rhythm_burden": round(irregular_rhythm_burden, 3),
                    "ecg_quality_score": round(ecg_quality_score, 3),
                    "valid_ecg_recordings": valid_ecg_recordings,
                    "deterioration_14d": deterioration_14d,
                }
            )

    return pd.DataFrame(rows)

print(generate_mock_data().head())
print(generate_mock_data().describe())
print(generate_mock_data().info())

def main() -> None:
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_mock_data()
    output_path = output_dir / "mock_wearable_clinical_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Mock dataset saved to: {output_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()