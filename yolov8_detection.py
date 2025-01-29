from ultralytics import YOLO
import cv2
import datetime
import os

def main():
    try:
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")  # nano version (fastest)
        
        # Initialize video capture
        print("Starting webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Set to store unique detections
        detected_objects = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame")
                break

            # Run YOLO detection
            results = model(frame, conf=0.5)  # Confidence threshold 0.5
            
            # Get current detections
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = result.names[class_id]
                    detected_objects.add(label)
            
            # Display frame with detections
            annotated_frame = results[0].plot()
            
            # Add detection count overlay
            current_detections = [f"{obj}" for obj in detected_objects]
            y_position = 30
            cv2.putText(annotated_frame, f"Objects Detected: {len(current_detections)}", 
                        (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("YOLOv8 Live Detection (Press 'q' to quit)", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save results
        if 'detected_objects' in locals() and detected_objects:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"yolov8_detections_{timestamp}.txt"
            
            with open(output_file, "w") as f:
                f.write("YOLOv8 Detected Objects:\n")
                for obj in sorted(detected_objects):
                    print(f"- {obj}")
                    f.write(f"- {obj}\n")
            
            print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    # Set environment variable for display
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    main()
