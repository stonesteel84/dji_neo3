import argparse
import cv2
from pathlib import Path

from .detectors import RepositoryYoloDetector

def main():
    parser = argparse.ArgumentParser(description='DJI Neo Tiny Object Detection')
    parser.add_argument('--weights', default=str(Path(__file__).parent / 'ffca_yolo/weights/best.pt'))
    parser.add_argument('--data', default=str(Path(__file__).parent / 'ffca_yolo/data/AITOD.yaml'))
    parser.add_argument('--source', default='0', help='Video source (0 for webcam, rtsp://... for stream)')
    parser.add_argument('--device', default='0', help='cuda device (0) or cpu')
    args = parser.parse_args()

    print("YOLO 모델을 불러오는 중입니다...")
    detector = RepositoryYoloDetector(
        weights=args.weights,
        data=args.data,
        target_labels=["person", "vehicle", "car"],
        device=args.device,
        confidence=0.2,
        iou=0.45
    )
    detector.load()
    print("모델 로드 완료!")

    # 정수 문자열이면 int로 변환 (웹캠)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        print(f"오류: 비디오 소스({src})를 열 수 없습니다.")
        return

    print(f"실시간 탐지를 시작합니다. (소스: {src}) 종료: 'q'")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 받아올 수 없습니다.")
            break

        detections = detector.detect(frame)

        for d in detections:
            x1, y1, x2, y2 = map(int, d.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"{d.label}:{d.confidence:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        cv2.imshow("DJI Neo - Tiny Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

