from ultralytics import YOLO
import cv2
import datetime
import os
import argparse

def process_video(video_path):
    try:
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        
        # Open video file
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_video_{timestamp}.mp4"
        output_writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, 
            (frame_width, frame_height)
        )

        detected_objects = set()
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"\rProcessing: {progress:.1f}% ({frame_count}/{total_frames})", end="")

            # Run YOLO detection
            results = model(frame, conf=0.5)
            
            # Get current detections
            for result in results:
                for box in result.boxes:
                    label = result.names[int(box.cls[0])]
                    detected_objects.add(label)
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Add processing info
            cv2.putText(annotated_frame, 
                       f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # Write frame to output video
            output_writer.write(annotated_frame)
            
            # Display frame (optional)
            cv2.imshow("Processing Video (Press 'q' to stop)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        cap.release()
        output_writer.release()
        cv2.destroyAllWindows()
        
        # Save detection results
        if 'detected_objects' in locals() and detected_objects:
            results_file = f"video_detections_{timestamp}.txt"
            
            with open(results_file, "w") as f:
                f.write(f"Video Analysis Results for: {video_path}\n")
                f.write(f"Processed frames: {frame_count}/{total_frames}\n")
                f.write("\nDetected Objects:\n")
                for obj in sorted(detected_objects):
                    print(f"- {obj}")
                    f.write(f"- {obj}\n")
            
            print(f"\nResults saved to: {results_file}")
            print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    # Set environment variable for display
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video file with YOLOv8')
    parser.add_argument('video_path', help='Path to the video file')
    args = parser.parse_args()
    
    process_video(args.video_path)
