import cv2
import os
import datetime
import numpy as np

# Force OpenCV to use X11 backend instead of Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"

required_files = {
    "weights": "yolov3-tiny.weights",
    "config": "yolov3-tiny.cfg",
    "names": "coco.names"
}

# Check if all required files exist
missing_files = []
for file_type, filename in required_files.items():
    if not os.path.isfile(filename):
        missing_files.append(filename)

if missing_files:
    print("Error: The following required files are missing:")
    for file in missing_files:
        print(f"- {file}")
    print("\nPlease download the files using these links:")
    print("- YOLOv3-tiny weights: https://pjreddie.com/media/files/yolov3-tiny.weights")
    print("- YOLOv3-tiny config: https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg")
    print("- COCO names: https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")
    exit(1)

# Load YOLO
net = cv2.dnn.readNet(required_files["weights"], required_files["config"])
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class labels
with open(required_files["names"], "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Add these constants after imports
CONFIDENCE_THRESHOLD = 0.4  # Lowered from 0.5
NMS_THRESHOLD = 0.3        # Non-maximum suppression threshold
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Start live video capture (0 for default webcam, change if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

detected_objects = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Initialize lists for detected objects
        class_ids = []
        confidences = []
        boxes = []

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = COLORS[class_ids[i]]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
                
                # Add to detected objects set
                detected_objects.add(label)

        # Display the live video
        cv2.imshow("Live Object Detection (Press 'q' to quit)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nDetection stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release resources and save results
    cap.release()
    cv2.destroyAllWindows()

    # Print detected objects
    print("Detected Objects:")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"detected_objects_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write("Detected Objects:\n")
        for obj in sorted(detected_objects):
            print(f"- {obj}")
            f.write(f"- {obj}\n")

    print(f"\nResults saved to: {output_file}")
