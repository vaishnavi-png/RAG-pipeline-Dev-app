df = pd.read_excel("/content/AIGTxt.xlsx")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df.dtypes)

df = df.drop(columns=["Unnamed: 4", "Unnamed: 5", "Unnamed: 6"])

# check if Mixed Text is genuinely different from ChatGPT-Generated
identical = (df["ChatGPT-Generated"] == df["Mixed Text"]).mean()
print(f"Rows where ChatGPT-Generated == Mixed Text: {identical:.1%}")

# null check
print(df.isnull().sum())

# domain distribution
print(df["Domain"].value_counts())

# rough length sanity check
df["human_len"] = df["Human-Generated"].astype(str).str.split().str.len()
df["ai_len"] = df["ChatGPT-Generated"].astype(str).str.split().str.len()
print(df[["human_len","ai_len"]].describe())
