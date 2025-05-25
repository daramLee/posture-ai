import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 1. 데이터 로드
df = pd.read_csv("data/processed/processed.csv")
X = df.drop('label', axis=1)
y = df['label']

# 2. 훈련/검증 나누기
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. 모델 학습
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. 성능 평가
y_pred = clf.predict(X_test)
print("📊 분류 결과:\n", classification_report(y_test, y_pred))
print("🧩 혼동 행렬:\n", confusion_matrix(y_test, y_pred))

# 5. 모델 저장
model_path = "models/posture_rf_model.joblib"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(clf, model_path)
print(f"✅ 모델 저장 완료: {model_path}")
