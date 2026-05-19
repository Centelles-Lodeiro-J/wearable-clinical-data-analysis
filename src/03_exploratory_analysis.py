import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def save_summary_table(df: pd.DataFrame) -> None:
    summary = df.groupby("deterioration_14d")[
        [
            "age",
            "comorbidity_count",
            "mean_heart_rate",
            "sdnn",
            "rmssd",
            "cough_count",
            "irregular_rhythm_burden",
            "ecg_quality_score",
        ]
    ].mean()

    output_path = Path("outputs/tables/group_summary.csv")
    summary.to_csv(output_path)

    print(f"Summary table saved to: {output_path}")


def plot_heart_rate_by_outcome(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))

    df.boxplot(
        column="mean_heart_rate",
        by="deterioration_14d"
    )

    plt.title("Mean heart rate by clinical deterioration outcome")
    plt.suptitle("")
    plt.xlabel("Clinical deterioration within 14 days")
    plt.ylabel("Mean heart rate, bpm")

    output_path = Path("outputs/figures/heart_rate_by_outcome.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


def plot_sdnn_by_outcome(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))

    df.boxplot(
        column="sdnn",
        by="deterioration_14d"
    )

    plt.title("Heart-rate variability by clinical deterioration outcome")
    plt.suptitle("")
    plt.xlabel("Clinical deterioration within 14 days")
    plt.ylabel("SDNN, ms")

    output_path = Path("outputs/figures/sdnn_by_outcome.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


def main() -> None:
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    input_path = Path("data/processed/cleaned_wearable_clinical_data.csv")
    df = pd.read_csv(input_path)

    save_summary_table(df)
    plot_heart_rate_by_outcome(df)
    plot_sdnn_by_outcome(df)


if __name__ == "__main__":
    main()