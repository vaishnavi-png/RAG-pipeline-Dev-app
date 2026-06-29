import pandas as pd

human_df = df[["Human-Generated", "Domain"]].rename(columns={"Human-Generated": "text"})
human_df["label"] = "human"

ai_df = df[["ChatGPT-Generated", "Domain"]].rename(columns={"ChatGPT-Generated": "text"})
ai_df["label"] = "ai"

# include Mixed Text only if you confirmed above it's actually distinct from ChatGPT-Generated
mixed_df = df[["Mixed Text", "Domain"]].rename(columns={"Mixed Text": "text"})
mixed_df["label"] = "mixed"

long_df = pd.concat([human_df, ai_df, mixed_df], ignore_index=True)
long_df = long_df.dropna(subset=["text"]).reset_index(drop=True)

print(long_df.shape)
print(long_df["label"].value_counts())
print(long_df.head())
