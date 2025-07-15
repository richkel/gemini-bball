"""
Simple webcam test script to verify camera functionality.
Displays the webcam feed with basic controls.
"""
import cv2
import sys

def test_webcam(camera_index=0, width=1280, height=720):
    """Test webcam feed with the specified camera index and resolution."""
    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera at index {camera_index}")
        print("Troubleshooting:")
        print("1. Make sure your camera is connected")
        print("2. Check if another application is using the camera")
        print("3. Try a different camera index (e.g., 1 or 2)")
        return False
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"🎥 Camera opened successfully!")
    print(f"📏 Resolution: {actual_width}x{actual_height}")
    print(f"📊 FPS: {fps:.2f}")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save a snapshot")
    print("\nPress any key to start the camera feed...")
    
    # Wait for a key press to start
    cv2.waitKey(1000)  # Small delay to show the message
    
    snapshot_count = 0
    
    try:
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
                
            # Display the frame
            cv2.imshow('Webcam Test (Press Q to quit)', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q' key
            if key == ord('q'):
                print("\n👋 Exiting...")
                break
                
            # Save snapshot on 's' key
            elif key == ord('s'):
                snapshot_count += 1
                filename = f'snapshot_{snapshot_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"📸 Snapshot saved as {filename}")
    
    except KeyboardInterrupt:
        print("\n👋 User interrupted the program")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera resources released")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test webcam functionality')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height (default: 720)')
    
    args = parser.parse_args()
    
    print(f"🔍 Testing camera at index {args.camera}...")
    success = test_webcam(args.camera, args.width, args.height)
    
    if success:
        print("✅ Webcam test completed successfully!")
    else:
        print("❌ Webcam test failed. Please check the error messages above.")
        sys.exit(1)
