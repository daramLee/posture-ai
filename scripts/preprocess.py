import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 전처리 결과와 기준을 저장.
# 전처리 결과와 기준은 훈련 데이터에 따라 달라짐. (실행할 때마다 달라진다는 의미.)
# 그래서 최종 훈련 데이터의 결과와 기준을 프로덕트에 사용해야 함.

# 0. 사용할 관절 번호 설정
landmark_indices = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]

# 1. 열 인덱스 계산 (x, y, z 각 관절당 3개)
cols = []
for i in landmark_indices:
    cols.extend([i * 3, i * 3 + 1, i * 3 + 2])

# 2. 원본 데이터 불러오기
df = pd.read_csv("data/posture_dataset_raw.csv")  # 전체 관절 데이터

# 3. 상체 관절만 추출
df = df.iloc[:, cols + [-1]]  # 끝에 라벨 포함되어 있다고 가정

# 🔥 잘못된 라벨 제거
df = df[df['label'].isin([0, 1, 2])]

# 4. 결측값 & 중복 제거
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 5. 라벨 분리
X = df.drop('label', axis=1)
y = df['label']

# 6. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 전처리 기준 저장 
joblib.dump(scaler, "models/scaler.pkl")

# 7. 라벨 분포 확인
print("📊 클래스 분포:\n", y.value_counts())

# 8. 결과 저장
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['label'] = y.values

output_path = "data/processed/processed.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
processed_df.to_csv(output_path, index=False)

print(f"✅ 전처리 및 저장 완료: {output_path}")
