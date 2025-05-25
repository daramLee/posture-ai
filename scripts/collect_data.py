import cv2
import mediapipe as mp
import pandas as pd
import os

# 동영상을 실행하고 좌표를 수집하는 코드

# MediaPipe 초기화
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 카메라 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

results = []

cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)

print("📷 자세 수집 중... q 키를 누르면 종료하고 저장됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = pose.process(rgb)

    if output.pose_landmarks:
        row = []
        for lm in output.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
        row.append(2)  # ✅ 좋은 자세면 1, 나쁜 자세면 2, unknown이면 0으로 수정하고나서 실행하기.
        results.append(row)

        # 관절 시각화
        mp_drawing.draw_landmarks(
            frame,
            output.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )

    cv2.imshow('Pose', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("⏹ 종료 요청됨.")
        break

cap.release()
cv2.destroyAllWindows()

# ✅ 저장할 파일 이름 입력받기
if len(results) > 0:
    filename = input("💾 저장할 파일 경로를 입력하세요 (예: data/good_posture.csv): ")

    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"✅ 데이터 저장 완료: {filename} ({len(results)}개 프레임)")
else:
    print("⚠️ 수집된 데이터가 없어 저장하지 않았습니다.")
