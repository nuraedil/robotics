from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Камера не открылась. Попробуйте другой индекс.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Не удалось получить кадр с камеры")
        break

    results = model(frame)[0]
    people_boxes = [box for box in results.boxes if int(box.cls) == 0]

    for box in people_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f'People Count: {len(people_boxes)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('YOLOv8 People Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
