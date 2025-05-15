# 프로젝트 설명
좋은 / 나쁜 자세를 판단하는 ai를 개발하는 프로젝트입니다.

# 프로젝트 구조

posture-ai/
├── data/                ← 수집한 자세 데이터 (CSV)
├── models/              ← 훈련된 모델 저장 (.pkl)
├── scripts/             ← 데이터 수집 및 전처리 코드
├── api/                 ← FastAPI 서버 코드
├── notebooks/           ← 테스트 및 실험용 Jupyter 노트북
├── posture_dataset.csv  ← 전처리된 전체 데이터
├── requirements.txt     ← 의존성 목록
└── README.md