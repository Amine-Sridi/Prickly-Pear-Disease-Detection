import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('datasets/runs/detect/train4/weights/last.pt')  # Replace with the path to your trained model

# Initialize the webcam (0 for default webcam, change if you have multiple webcams)
cap = cv2.VideoCapture(1)

# Define a function to perform real-time detection
def run_real_time_detection():
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Perform inference with the YOLO model
        results = model.predict(source=frame, show=True)  # Set show=True if you want YOLO's display

        # Process results
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates, class, and confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                predicted_class = int(box.cls)  # Predicted class index
                confidence = box.conf  # Confidence score

                # Map predicted class index to class name (adjust as necessary)
                class_names = ['Healthy', 'Diseased']
                class_name = class_names[predicted_class]

                # Draw the bounding box and label on the frame
                color = (0, 255, 0) if class_name == 'Healthy' else (0, 0, 255)  # Green for Healthy, Red for Diseased
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Correct conversion of confidence from tensor to float
                label = f"{class_name} ({confidence.item():.2f})"

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time detection function
run_real_time_detection()
