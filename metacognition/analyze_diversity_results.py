import pandas as pd
import json

with open("metacognition/outputs/diversity_results.json", "r") as fp:
    results = json.load(fp)

for result in results:
    result["baseline_bug_category"] = result["baseline"]["bug_category"]
    result["baseline_modified_lines"] = result["baseline"]["modified_lines"]
    result["baseline_passes_test_case"] = result["baseline"]["passes_test_case"]

    result["exemplar_bug_category"] = result["exemplar"]["bug_category"]
    result["exemplar_modified_lines"] = result["exemplar"]["modified_lines"]
    result["exemplar_passes_test_case"] = result["exemplar"]["passes_test_case"]

df = pd.DataFrame(results)
df = df[["baseline_bug_category", "baseline_modified_lines", "baseline_passes_test_case",
         "exemplar_bug_category", "exemplar_modified_lines", "exemplar_passes_test_case",
         "sample_path", "solution_index"]]

print(df.groupby("sample_path")[["baseline_bug_category"]].value_counts())
print()
print(df.groupby("sample_path")[["exemplar_bug_category"]].value_counts())
print("========================")

print(df.groupby("sample_path")[["baseline_passes_test_case"]].value_counts())
print()
print(df.groupby("sample_path")[["exemplar_passes_test_case"]].value_counts())
print("========================")

def value_counts_for_lists(series):
    flattened = [item for sublist in series for item in sublist]
    return pd.Series(flattened).value_counts()

print(df.groupby('sample_path')['baseline_modified_lines'].apply(value_counts_for_lists).unstack())
print()
print(df.groupby('sample_path')['exemplar_modified_lines'].apply(value_counts_for_lists).unstack())
