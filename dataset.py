DATA_PATH = "Cardiac_Dataset.xlsx"

df = pd.read_excel(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:")
print(df.columns.tolist())
print(df.head())
print(df.isnull().sum())