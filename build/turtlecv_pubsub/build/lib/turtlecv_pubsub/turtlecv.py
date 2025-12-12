#!/usr/bin/env python3
"""
Mock Human Guide Node with Simulated Navigation and Depth Sensing
No OpenCV, TurtleBot, or Nav2 required - everything is mocked!
Fixed: Removed info_throttle for compatibility with all ROS2 versions
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
import random
import time
import math

class MockDepthCamera(Node):
    """Mock depth camera that publishes simulated depth data"""
    
    def __init__(self):
        super().__init__("mock_depth_camera")
        
        # Publishers
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.distance_pub = self.create_publisher(Float32, '/obstacle_distance', 10)
        
        # Timer for publishing depth data (30 Hz)
        self.timer = self.create_timer(0.033, self.publish_depth_data)
        
        # Simulation state
        self.time_elapsed = 0.0
        self.obstacle_scenario = 0
        
        self.get_logger().info("Mock Depth Camera Started - Publishing simulated depth data")
    
    def publish_depth_data(self):
        """Publish simulated depth readings"""
        self.time_elapsed += 0.033
        
        # Simulate different obstacle scenarios over time
        if self.time_elapsed < 5.0:
            # Clear path
            distance = random.uniform(3.0, 5.0)
        elif self.time_elapsed < 10.0:
            # Approaching obstacle
            distance = 2.0 - (self.time_elapsed - 5.0) * 0.3
            distance = max(0.3, distance)
        elif self.time_elapsed < 15.0:
            # Very close obstacle
            distance = random.uniform(0.2, 0.4)
        elif self.time_elapsed < 20.0:
            # Obstacle moving away
            distance = 0.4 + (self.time_elapsed - 15.0) * 0.4
        else:
            # Reset cycle
            self.time_elapsed = 0.0
            distance = random.uniform(3.0, 5.0)
        
        # Add some noise
        distance += random.uniform(-0.1, 0.1)
        distance = max(0.1, distance)  # Minimum 10cm
        
        # Publish distance
        dist_msg = Float32()
        dist_msg.data = distance
        self.distance_pub.publish(dist_msg)
        
        # Mock Image message (we don't need actual image data)
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_depth_frame"
        img_msg.height = 480
        img_msg.width = 640
        self.depth_pub.publish(img_msg)


class MockNavigationStack(Node):
    """Mock Nav2 navigation - simulates robot movement"""
    
    def __init__(self):
        super().__init__("mock_navigation")
        
        # Current robot pose
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        
        # Goal tracking
        self.goal_x = None
        self.goal_y = None
        self.has_goal = False
        self.is_moving = False
        
        # Throttling for logging
        self.last_log_time = 0.0
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/current_pose', 10)
        self.status_pub = self.create_publisher(String, '/nav_status', 10)
        
        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # Timer for simulating movement
        self.timer = self.create_timer(0.1, self.update_position)
        
        self.get_logger().info("Mock Navigation Stack Started")
    
    def goal_callback(self, msg):
        """Receive new navigation goal"""
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True
        self.is_moving = True
        
        self.get_logger().info(f"New goal received: ({self.goal_x:.2f}, {self.goal_y:.2f})")
        
        status = String()
        status.data = "Goal accepted - navigating"
        self.status_pub.publish(status)
    
    def update_position(self):
        """Simulate robot moving toward goal"""
        if not self.has_goal or not self.is_moving:
            return
        
        # Calculate distance to goal
        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if reached goal
        if distance < 0.1:  # Within 10cm
            self.has_goal = False
            self.is_moving = False
            self.get_logger().info(f"✓ Reached goal at ({self.current_x:.2f}, {self.current_y:.2f})")
            
            status = String()
            status.data = "Goal reached"
            self.status_pub.publish(status)
            return
        
        # Move toward goal (simulate 0.2 m/s speed)
        speed = 0.02  # 0.2 m/s * 0.1s update rate
        move_distance = min(speed, distance)
        
        self.current_x += (dx / distance) * move_distance
        self.current_y += (dy / distance) * move_distance
        
        # Publish current pose
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = self.current_x
        pose.pose.position.y = self.current_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        
        # Manual throttled logging (1 second interval)
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_log_time >= 1.0:
            self.get_logger().info(
                f"Moving: pos=({self.current_x:.2f}, {self.current_y:.2f}), "
                f"distance to goal={distance:.2f}m"
            )
            self.last_log_time = current_time


class MockObstacleAvoidance(Node):
    """Mock obstacle avoidance with velocity control"""
    
    def __init__(self):
        super().__init__("mock_obstacle_avoidance")
        
        # Parameters
        self.safe_distance = 0.5  # meters
        self.max_speed = 0.22  # m/s
        self.current_distance = 10.0
        
        # Throttling for logging
        self.last_log_time = 0.0
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        
        # Subscribers
        self.distance_sub = self.create_subscription(
            Float32,
            '/obstacle_distance',
            self.distance_callback,
            10
        )
        
        self.get_logger().info("Mock Obstacle Avoidance Started")
    
    def distance_callback(self, msg):
        """Process obstacle distance and adjust velocity"""
        self.current_distance = msg.data
        
        # Calculate appropriate velocity
        twist = Twist()
        status_msg = String()
        
        if self.current_distance < self.safe_distance * 0.5:
            # STOP - too close!
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # Turn to avoid
            status_msg.data = f"⚠️ OBSTACLE TOO CLOSE ({self.current_distance:.2f}m) - STOPPING"
            self.get_logger().warn(status_msg.data)
            
        elif self.current_distance < self.safe_distance:
            # Slow down
            speed_factor = (self.current_distance - self.safe_distance * 0.5) / (self.safe_distance * 0.5)
            twist.linear.x = self.max_speed * speed_factor
            twist.angular.z = 0.0
            status_msg.data = f"⚡ Obstacle detected ({self.current_distance:.2f}m) - SLOW ({twist.linear.x:.2f} m/s)"
            self.get_logger().info(status_msg.data)
            
        else:
            # Full speed ahead - with manual throttling
            twist.linear.x = self.max_speed
            twist.angular.z = 0.0
            status_msg.data = f"✓ Path clear ({self.current_distance:.2f}m) - NORMAL SPEED"
            
            # Manual throttled logging (2 second interval)
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.last_log_time >= 2.0:
                self.get_logger().info(status_msg.data)
                self.last_log_time = current_time
        
        # Publish commands
        self.cmd_vel_pub.publish(twist)
        self.safety_status_pub.publish(status_msg)


class MockHumanGuideNode(Node):
    """Mock human guidance system - coordinates navigation"""
    
    def __init__(self):
        super().__init__("mock_human_guide")
        
        # Define waypoints (Point A and Point B)
        self.point_A = (2.0, 3.0)
        self.point_B = (5.0, 1.0)
        self.current_waypoint = 0
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.guide_status_pub = self.create_publisher(String, '/guide_status', 10)
        
        # Subscribers
        self.nav_status_sub = self.create_subscription(
            String,
            '/nav_status',
            self.nav_status_callback,
            10
        )
        
        # Start guidance after a short delay
        self.timer = self.create_timer(2.0, self.start_guidance)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Mock Human Guide System Started")
        self.get_logger().info(f"Point A: {self.point_A}")
        self.get_logger().info(f"Point B: {self.point_B}")
        self.get_logger().info("=" * 60)
    
    def start_guidance(self):
        """Start the guidance sequence"""
        self.timer.cancel()  # Stop the startup timer
        
        self.get_logger().info("🚀 Starting human guidance: Heading to Point A")
        self.send_goal(self.point_A[0], self.point_A[1])
        
        status = String()
        status.data = "Guidance started - heading to Point A"
        self.guide_status_pub.publish(status)
    
    def send_goal(self, x, y):
        """Send navigation goal"""
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        
        self.goal_pub.publish(goal)
        self.get_logger().info(f"📍 Goal sent: ({x:.2f}, {y:.2f})")
    
    def nav_status_callback(self, msg):
        """Handle navigation status updates"""
        if "Goal reached" in msg.data:
            if self.current_waypoint == 0:
                # Reached Point A, go to Point B
                self.current_waypoint = 1
                self.get_logger().info("=" * 60)
                self.get_logger().info("✓ Reached Point A!")
                self.get_logger().info("🚀 Now heading to Point B")
                self.get_logger().info("=" * 60)
                
                # Wait a moment before next waypoint
                time.sleep(1.0)
                self.send_goal(self.point_B[0], self.point_B[1])
                
                status = String()
                status.data = "At Point A - heading to Point B"
                self.guide_status_pub.publish(status)
                
            elif self.current_waypoint == 1:
                # Reached Point B, guidance complete
                self.get_logger().info("=" * 60)
                self.get_logger().info("🎉 Reached Point B - Guidance Complete!")
                self.get_logger().info("=" * 60)
                
                status = String()
                status.data = "Guidance complete - arrived at Point B"
                self.guide_status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    
    # Create all mock nodes
    depth_camera = MockDepthCamera()
    navigation = MockNavigationStack()
    obstacle_avoidance = MockObstacleAvoidance()
    guide = MockHumanGuideNode()
    
    # Create executor to run all nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(depth_camera)
    executor.add_node(navigation)
    executor.add_node(obstacle_avoidance)
    executor.add_node(guide)
    
    try:
        print("\n" + "=" * 60)
        print("MOCK HUMAN GUIDE SYSTEM - ALL SYSTEMS RUNNING")
        print("=" * 60)
        print("Components:")
        print("  ✓ Mock Depth Camera (publishing distance data)")
        print("  ✓ Mock Navigation Stack (simulating movement)")
        print("  ✓ Mock Obstacle Avoidance (velocity control)")
        print("  ✓ Mock Human Guide (waypoint coordination)")
        print("=" * 60)
        print("Watch the robot navigate from (0,0) → Point A → Point B")
        print("Obstacles will appear randomly to test avoidance!")
        print("=" * 60 + "\n")
        
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        depth_camera.destroy_node()
        navigation.destroy_node()
        obstacle_avoidance.destroy_node()
        guide.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
