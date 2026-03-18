import pickle
import pandas as pd

path = "/data/nas/users/ruijie.sang/logs/wmnav_qwen_train1/ObjectNav_wmnav-Qwen2_5-VL-7B-Instruct-hm3dv2/0_of_50/0_877/df_results.pkl"

with open(path, "rb") as f:
    df = pickle.load(f)

print("Type:", type(df))
if isinstance(df, pd.DataFrame):
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print(df.head(10))
elif isinstance(df, dict):
    print("Keys:", list(df.keys()))
    for k, v in df.items():
        print(f"  {k}: {v}")
else:
    print(df)