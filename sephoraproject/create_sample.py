import pandas as pd

df = pd.read_csv("reviews_0-250.csv", low_memory=False)

sample_df = df.sample(n=20000, random_state=42)

sample_df.to_csv("reviews_sample.csv", index=False)

print("Done!")