import cv2

cap = cv2.VideoCapture(0)  # Webcam
ret, frame = cap.read()
cap.release()

cv2.imshow("Frame", frame)
cv2.waitKey(1)

roi = cv2.selectROI("Pilih Area", frame, fromCenter=False, showCrosshair=True)
print(f"ROI dipilih: {roi}")
cv2.destroyAllWindows()
