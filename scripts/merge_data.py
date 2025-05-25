import pandas as pd
import glob
import os

# 각 데이터 파일의 좌표를 합치는 코드

label_folders = ['good', 'bad', 'unknown']
dfs = []

for label_name in label_folders:
    folder_path = f"data/raw/{label_name}"
    files = glob.glob(os.path.join(folder_path, "*.csv"))

    for f in files:
        df = pd.read_csv(f, header=None)  # 🔥 컬럼 이름 없음 처리
        df.columns = [f"x{i//3}" if i % 3 == 0 else (f"y{i//3}" if i % 3 == 1 else f"z{i//3}") for i in range(df.shape[1] - 1)] + ["label"]
        dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv("data/posture_dataset_raw.csv", index=False)

print(f"✅ 총 데이터: {len(merged)}")
