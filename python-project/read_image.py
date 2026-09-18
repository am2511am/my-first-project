import cv2

# 1. โหลด Haar Cascade XML
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 2. เปิดใช้งาน Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถดึงภาพจากกล้องได้")
        break

    # 3. แปลงเป็น Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 5. วาดกรอบสี่เหลี่ยมรอบทุกใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 6. แสดงจำนวนใบหน้า (Workshop Challenge)
    cv2.putText(
        frame, 
        f"Faces: {len(faces)}", 
        (20, 40), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (0, 255, 0), 
        2
    )

    # แสดงผล
    cv2.imshow("Student Face Detector", frame)

    # กดปุ่ม 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()