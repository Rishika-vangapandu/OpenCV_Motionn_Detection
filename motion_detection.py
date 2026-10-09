import cv2

# Open the MacBook webcam
cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Camera not available!")
    exit()

# Read the first frame
ret, first_frame = cap.read()

if not ret:
    print("Could not read camera frame!")
    cap.release()
    exit()

# Convert first frame to grayscale and blur it
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compare current frame with the first frame
    difference = cv2.absdiff(first_gray, gray)

    # Highlight changed areas
    _, threshold = cv2.threshold(
        difference, 25, 255, cv2.THRESH_BINARY
    )

    # Find moving regions
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame, "Motion detected", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("Motion Detection", frame)
     #cv2.imshow("Motion Mask", threshold)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
difference = cv2.absdiff(first_gray, gray)

difference = cv2.GaussianBlur(
    difference, (21, 21), 0
)

_, threshold = cv2.threshold(
    difference, 30, 255, cv2.THRESH_BINARY
)

threshold = cv2.dilate(
    threshold, None, iterations=2
)