import pandas as pd
from pathlib import Path
from tableone import TableOne
from ydata_profiling import ProfileReport


# Define a funtion to clean the data
input_path = Path("data/raw/mock_wearable_clinical_data.csv")
df = pd.read_csv(input_path)

# Ydata profiling report for the raw data before cleaning and preprocessing to understand the data and identify any issues or patterns that may need to be addressed in the cleaning process. 
profile = ProfileReport(
    df,
    title="Mock wearable clinical data profiling report",
    explorative=True,
)

output_dir = Path("outputs/reports")
output_dir.mkdir(parents=True, exist_ok=True)

profile.to_file(output_dir / "raw_data_profile_report.html")

# Tableone summary table by deterioration status to compare the characteristics of patients who deteriorated within 14 days and those who did not. This can help identify any differences in demographics, comorbidities, or other factors that may be associated with deterioration. The table will include p-values to assess the statistical significance of any observed differences between the groups, and it will also report the number of missing values for each variable to provide insight into the completeness of the data.
columns = [
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

categorical = [
    "sex",
    "prior_hospitalisation",
]

groupby = "deterioration_14d"

table = TableOne(
    data=df,
    columns=columns,
    categorical=categorical,
    groupby=groupby,
    pval=True,
    missing=True,
)

print(table.tabulate(tablefmt="github"))

output_dir = Path("outputs/tables")
output_dir.mkdir(parents=True, exist_ok=True)

table.to_csv(output_dir / "tableone_raw_by_deterioration.csv")