import pandas as pd

df = pd.read_csv("unsafe_act_data.csv")

print("Original SIF Binary balance:")
print(df["sif_binary"].value_counts())

# Find the smallest class
min_count = df["sif_binary"].value_counts().min()
print(f"\nSmallest class has {min_count} reports")

# Sample equal number from each class
balanced_list = []
for value in [0, 1]:
    subset = df[df["sif_binary"] == value]
    sampled = subset.sample(n=min_count, random_state=42)
    balanced_list.append(sampled)

balanced_df = pd.concat(balanced_list, ignore_index=True)

print("\nBalanced dataset:")
print(balanced_df["sif_binary"].value_counts())

# Save balanced dataset
balanced_df.to_csv("unsafe_act_balanced.csv", index=False)
print(f"\n✅ Saved 'unsafe_act_balanced.csv' with {len(balanced_df)} reports")
