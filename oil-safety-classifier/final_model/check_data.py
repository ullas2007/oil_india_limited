import pandas as pd

df = pd.read_csv("unsafe_act_data.csv")

print("=" * 60)
print("DATASET ANALYSIS")
print("=" * 60)
print(f"Total reports: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nSIF Binary Distribution:")
print(df["sif_binary"].value_counts())
print("\nLabel Source Distribution:")
print(df["label_source"].value_counts())
print("\nSample reports:")
print(df[["report_text", "sif_binary"]].head(3))
