#!/usr/bin/env python3
"""
Pink Human Detector for ROS2 + TurtleBot3
Fixed: Safe camera handling to prevent bus errors
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Try to import YOLO, fallback if not available
'''
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("\u26a0\ufe0f  YOLO not available. Install with: pip install ultralytics")
'''
YOLO_AVAILABLE = False
class PinkDetectorNode(Node):
    def __init__(self):
        super().__init__('pink_detector')
        
        # OpenCV Bridge
        self.bridge = CvBridge()
        
        # Detection counters
        self.frame_count = 0
        self.detection_count = 0
        self.camera_available = False
        
        # Robot control
        self.auto_mode = False
        self.pink_detected = False
        self.pink_center_x = None
        
        # YOLO model (optional)
        self.model = None
        if YOLO_AVAILABLE:
            self.get_logger().info('\U0001f504 Loading YOLO model...')
            try:
                self.model = YOLO('yolov8n.pt')
                self.get_logger().info('\u2705 YOLO model loaded successfully!')
            except Exception as e:
                self.get_logger().warn(f'\u26a0\ufe0f  YOLO not available: {e}')
                self.get_logger().info('Running in color-detection-only mode')
        else:
            self.get_logger().info('Running in color-detection-only mode (no YOLO)')
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.detection_pub = self.create_publisher(String, '/pink_detection', 10)
        self.image_pub = self.create_publisher(Image, '/pink_detected_image', 10)
        
        # Subscribers
        self.camera_sub = self.create_subscription(
            Image, 
            '/image_raw', 
            self.camera_callback, 
            10
        )
        
        # Timeout timer to check if camera is publishing
        self.camera_timeout = self.create_timer(5.0, self.check_camera_status)
        self.last_frame_time = None
        
        # Control timer (for auto mode)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('=== Pink Human Detector Started ===')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Subscribed to: /camera/image_raw')
        self.get_logger().info('Publishing to: /cmd_vel, /pink_detection, /pink_detected_image')
        self.get_logger().info('')
        self.get_logger().info('Waiting for camera feed...')
        self.get_logger().info('If no camera, the node will run in simulation mode')
        self.get_logger().info('=' * 60)

    def check_camera_status(self):
        """Check if camera feed is being received"""
        if not self.camera_available:
            self.get_logger().warn('\u26a0\ufe0f  No camera feed detected on /camera/image_raw')
            self.get_logger().info('\U0001f4a1 Available options:')
            self.get_logger().info('   1. Connect a camera or start camera node')
            self.get_logger().info('   2. Use simulation: ros2 run turtlecv_pubsub turtlecv')
            self.get_logger().info('   3. Check available topics: ros2 topic list')

    def detect_pink_color_only(self, img):
        """
        Detect pink color without YOLO (fallback method)
        """
        self.frame_count += 1
        h, w = img.shape[:2]
        
        frame = img.copy()
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Pink color range
        mask1 = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([20, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([150, 15, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_pink = False
        pink_positions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                found_pink = True
                self.detection_count += 1
                
                x, y, w_c, h_c = cv2.boundingRect(contour)
                cx = x + w_c // 2
                pink_positions.append(cx)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w_c, y + h_c), (255, 0, 255), 3)
                cv2.putText(frame, f"PINK {area:.0f}px", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Draw center
                cv2.circle(frame, (cx, y + h_c//2), 10, (255, 0, 255), -1)
        
        # Status overlay
        status = f"Frame:{self.frame_count} | Pink:{self.detection_count}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.auto_mode:
            cv2.putText(frame, "AUTO MODE ON", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if found_pink:
            cv2.putText(frame, "PINK DETECTED!", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
            pink_center = int(np.mean(pink_positions)) if pink_positions else None
        else:
            pink_center = None
        
        # Draw center line
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        
        return frame, found_pink, pink_center

    def detect_pink_humans(self, img):
        """
        Detect humans wearing pink clothes using YOLO + Color detection
        Falls back to color-only detection if YOLO unavailable
        """
        if not self.model:
            return self.detect_pink_color_only(img)
        
        self.frame_count += 1
        h, w = img.shape[:2]
        
        # YOLO detection
        results = self.model(img, verbose=False, conf=0.15)
        
        frame = img.copy()
        found_pink = False
        pink_positions = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # person class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Extract ROI
                    roi = img[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # Color detection in HSV space
                    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Pink color range
                    mask1 = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([20, 255, 255]))
                    mask2 = cv2.inRange(hsv, np.array([150, 15, 50]), np.array([180, 255, 255]))
                    mask = cv2.bitwise_or(mask1, mask2)
                    
                    # Calculate pink ratio
                    pink_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])
                    
                    # If more than 2% is pink
                    if pink_ratio > 0.02:
                        color = (255, 0, 255)
                        label = f"PINK {pink_ratio*100:.0f}%"
                        thickness = 5
                        found_pink = True
                        self.detection_count += 1
                        
                        cx = (x1 + x2) // 2
                        pink_positions.append(cx)
                        
                        cv2.circle(frame, (cx, (y1+y2)//2), 10, (255, 0, 255), -1)
                    else:
                        color = (0, 255, 255)
                        label = f"Person {conf:.2f}"
                        thickness = 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Status overlay
        status = f"Frame:{self.frame_count} | Pink:{self.detection_count}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if self.auto_mode:
            cv2.putText(frame, "AUTO MODE ON", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        if found_pink:
            cv2.putText(frame, "PINK DETECTED!", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
            pink_center = int(np.mean(pink_positions)) if pink_positions else None
        else:
            pink_center = None
        
        cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 0), 2)
        
        return frame, found_pink, pink_center

    def camera_callback(self, msg: Image):
        """Process camera feed and detect pink humans - SAFE VERSION"""
        try:
            # Mark camera as available
            if not self.camera_available:
                self.camera_available = True
                self.get_logger().info('\u2705 Camera feed connected!')
            
            self.last_frame_time = self.get_clock().now()
            
            # Convert ROS Image to OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Detect pink humans
            processed_frame, found_pink, pink_center = self.detect_pink_humans(frame)
            
            # Update detection state
            self.pink_detected = found_pink
            self.pink_center_x = pink_center
            
            # Display
            cv2.imshow("Pink Human Detection", processed_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):
                self.auto_mode = not self.auto_mode
                mode = "ENABLED" if self.auto_mode else "DISABLED"
                self.get_logger().info(f'\U0001f916 AUTO mode {mode}')
                if not self.auto_mode:
                    self.stop_robot()
            elif key == ord('q'):
                self.get_logger().info('Shutting down...')
                rclpy.shutdown()
            
            # Publish detection result
            detection_msg = String()
            detection_msg.data = f"Pink detected: {found_pink}, Count: {self.detection_count}"
            self.detection_pub.publish(detection_msg)
            
            # Publish processed image
            img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
            self.image_pub.publish(img_msg)
            
            if found_pink:
                self.get_logger().info(f'\U0001f389 PINK DETECTED at x={pink_center}! (Frame {self.frame_count})')
            
        except Exception as e:
            self.get_logger().error(f'\u274c Camera callback error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def control_loop(self):
        """Control robot based on pink detection (AUTO mode)"""
        if not self.auto_mode or not self.camera_available:
            return
        
        if self.pink_detected and self.pink_center_x is not None:
            # Calculate error from center
            frame_center = 320
            error = self.pink_center_x - frame_center
            
            # Proportional control
            angular_z = -float(error) / 200.0
            
            # Move forward while tracking
            self.move_robot(0.1, angular_z)
        else:
            # No pink detected - rotate to search
            self.move_robot(0.0, 0.3)

    def move_robot(self, linear_x, angular_z):
        """Send velocity command to robot"""
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):
        """Stop the robot"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info('\U0001f6d1 Robot stopped')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PinkDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    except Exception as e:
        print(f"Error starting node: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            node.stop_robot()
            cv2.destroyAllWindows()
            node.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()
