import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math
import time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav2_msgs.action import NavigateToPose
import cv2
import numpy as np
from cv_bridge import CvBridge

class Nav2Explorer(Node):
    def __init__(self):
        super().__init__('nav2_explorer')
        
        # Navigation action client connects to Nav2
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Camera subscriber receives image data for visual processing
        self._image_sub = self.create_subscription(
            Image, '/camera/image_raw', self._image_callback, 10)
        # Laser scan subscriber provides distance measurements
        self._scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
            
        # Command velocity publisher controls robot movement
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Image processing tools setup
        self._bridge = CvBridge()
        # Debug window for viewing camera feed
        cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Feed', 500, 300)
        
        # Navigation state tracking
        self._nav_complete = False  # Flag for navigation completion
        self._goal_handle = None    # Active navigation goal reference
        
        # Behavior control flags
        self._should_spin = False          # 360 degree scan active flag
        self._should_approach = False      # Box approach active flag
        self._spin_start_time = None       # Timestamp when spin began
        self._approach_complete = False    # Successful approach completion
        self._blue_detected_during_spin = False  # Blue detection during scan
        self._mission_complete = False     # Final mission completion state
        
        # Object detection state variables
        self._blue_box_detected = False    # Blue box presence state
        self._green_box_detected = False   # Green box presence state
        self._red_box_detected = False     # Red box presence state
        self._x_offset = 0.0              # Horizontal offset of detected object
        self._box_area = 0.0              # Size of detected object in pixels
        self._distance_ahead = None        # Forward obstacle distance reading
        
        # Movement control parameters
        self._forward_speed = 0.15         # Base forward movement speed
        self._angular_kp = 0.003           # Steering correction gain
        self._stop_distance = 0.5          # Target stopping distance
        self._offset_threshold = 40.0      # Threshold for turns

    # Initiate navigation to specified map coordinates
    def navigate_to(self, x, y, yaw):
        if self._mission_complete:
            return
            
        # Reset navigation state flags
        self._nav_complete = False
        self._approach_complete = False
        self._blue_detected_during_spin = False
        
        # Configure navigation goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        
        # Convert yaw to quaternion orientation
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2)
        
        # Send navigation goal and setup response handler
        self._nav_client.wait_for_server()
        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._nav_response_callback)

	# Initiates 360 degree scan
    def start_spin(self):
        if self._mission_complete:
            return
            
        self._should_spin = True
        self._spin_start_time = time.time()
        self._blue_detected_during_spin = False

	# Begins box approach behavior
    def start_approach(self):
        if self._mission_complete:
            return
            
        self._should_approach = True
        self._approach_complete = False
        self.get_logger().info("Starting approach to blue box!")
    # Controls robot movement during 360 degree scan
    def _execute_spin(self):
        if self._mission_complete:
            self._cmd_vel_pub.publish(Twist())
            self._should_spin = False
            return
            
        # Continue spinning for 4 seconds
        if time.time() - self._spin_start_time < 4.0:
            twist = Twist()
            twist.angular.z = 1.57  # ~90 degrees per second rotation
            
            # Monitor for blue box detections during spin
            if self._blue_box_detected:
                self._blue_detected_during_spin = True
                self.get_logger().info("Blue box detected during scan")
                
            self._cmd_vel_pub.publish(twist)
        else:
            # Stop spinning when time completes
            self._cmd_vel_pub.publish(Twist())
            self._should_spin = False
            
            # Begin approach if blue was detected
            if self._blue_detected_during_spin:
                self.start_approach()
    # Handles movement control during box approach
    def _execute_approach(self):
        if self._mission_complete:
            self._cmd_vel_pub.publish(Twist())
            self._should_approach = False
            return
            
        # Stop when within target distance of object
        if self._distance_ahead is not None and self._distance_ahead <= self._stop_distance:
            self._cmd_vel_pub.publish(Twist())
            self._should_approach = False
            self._approach_complete = True
            self._mission_complete = True
            self.get_logger().info("Mission complete - Stopped at blue box!")
            return
            
        # Calculate movement commands based on object position
        twist = Twist()
        offset_abs = abs(self._x_offset)
        
        # Prioritize turning when object is significantly off-center
        if offset_abs > self._offset_threshold:
            twist.angular.z = -self._angular_kp * self._x_offset * 2.0
        else:
            # Move forward with minor steering corrections
            twist.linear.x = self._forward_speed
            twist.angular.z = -self._angular_kp * self._x_offset
            
        self._cmd_vel_pub.publish(twist)
    # Processes incoming camera images for color detection
    def _image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            # Display image in debug window
            cv2.imshow('Camera Feed', cv_image)
            cv2.waitKey(1)
            
            # Convert to HSV color space for better color isolation
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Color threshold ranges for box detection
            sensitivity = 10
            blue_lower = np.array([110 - sensitivity, 100, 100])
            blue_upper = np.array([110 + sensitivity, 255, 255])
            green_lower = np.array([60 - sensitivity, 100, 100])
            green_upper = np.array([60 + sensitivity, 255, 255])
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([sensitivity, 255, 255])
            red_lower2 = np.array([180 - sensitivity, 100, 100])
            red_upper2 = np.array([180, 255, 255])
            
            # Create color masks for detection
            blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
            green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
            red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Process each color mask for object detection
            self._detect_box(blue_mask, 'blue')
            self._detect_box(green_mask, 'green')
            self._detect_box(red_mask, 'red')
            
            # Handle immediate blue box detection scenarios
            if self._blue_box_detected and not self._mission_complete and not self._should_approach:
                if self._goal_handle is not None:
                    cancel_future = self._goal_handle.cancel_goal_async()
                    cancel_future.add_done_callback(self._cancel_done)
                self.start_approach()
            
        except Exception as e:
            self.get_logger().error(f"Image processing error: {e}")
    #Detects colored boxes in image mask
    def _detect_box(self, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Reset detection state if no contours found
            if color == 'blue':
                self._blue_box_detected = False
            elif color == 'green':
                self._green_box_detected = False
            elif color == 'red':
                self._red_box_detected = False
            return
            
        # Find largest contour in mask
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Only process detections above minimum size threshold
        if area > 30.0:
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                img_center = mask.shape[1] // 2
                
                # Update detection state based on color
                if color == 'blue':
                    self._x_offset = cx - img_center
                    self._box_area = area
                    self._blue_box_detected = True
                    self.get_logger().info(f"Blue box detected (Area: {area:.1f}, Offset: {self._x_offset:.1f})")
                elif color == 'green':
                    self._green_box_detected = True
                    self.get_logger().info(f"Green box detected (Area: {area:.1f})")
                elif color == 'red':
                    self._red_box_detected = True
                    self.get_logger().info(f"Red box detected (Area: {area:.1f})")
    #Processes LIDAR scan data for obstacle detection
    def _scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        # Use center 5 readings for forward distance measurement
        front_ranges = ranges[0:5]
        # Filter out invalid readings
        valid = [r for r in front_ranges if 0.2 < r < 25.0]
        # Calculate average forward distance
        self._distance_ahead = sum(valid) / len(valid) if valid else None
    # Handles response from navigation goal submission
    def _nav_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected!")
            self._nav_complete = True
            return
            
        self.get_logger().info("Navigation goal accepted")
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_callback)
    # Processes final navigation result
    def _nav_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info("Reached waypoint!")
            self._nav_complete = True
        else:
            self.get_logger().warn(f"Navigation failed with status: {status}")
            self._nav_complete = True

    def _cancel_done(self, future):
        self.get_logger().info("Navigation goal canceled for blue box approach")
        self._goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    explorer = Nav2Explorer()
    
    # Navigation waypoints
    waypoints = [
        (3.0, -8.27, -0.00143),    # First navigation target
        (-7.73, -6.77, -0.00143),   # Second navigation target
        (7.27, 5.25, -0.00143),      # Third navigation target
        (0.144, -0.00, 0.0124)      # Fourth navigation target
    ]
    current_waypoint = 0
    
    try:
        # Begin navigation sequence at first waypoint
        explorer.navigate_to(*waypoints[current_waypoint])
        
        while rclpy.ok():
            rclpy.spin_once(explorer, timeout_sec=0.1)
            
            # Skip behavior logic if mission is complete
            if explorer._mission_complete:
                continue
                
            # Handle navigation completion transitions
            if explorer._nav_complete and not explorer._approach_complete and not explorer._should_spin:
                explorer.start_spin()
            
            # Execute spin behavior when active
            if explorer._should_spin:
                explorer._execute_spin()
                
                # Determine next action after spin completes
                if not explorer._should_spin and not explorer._mission_complete:
                    if explorer._blue_detected_during_spin:
                        explorer.start_approach()
                    else:
                        current_waypoint += 1
                        if current_waypoint < len(waypoints):
                            explorer.navigate_to(*waypoints[current_waypoint])
                        else:
                            explorer.get_logger().info("All waypoints visited - Mission complete")
                            explorer._mission_complete = True
            
            # Execute approach behavior when active
            if explorer._should_approach:
                explorer._execute_approach()

            cv2.waitKey(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        explorer._cmd_vel_pub.publish(Twist())  # Stop all movement
        explorer.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()