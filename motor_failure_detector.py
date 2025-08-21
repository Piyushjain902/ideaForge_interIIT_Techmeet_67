import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from px4_msgs.msg import SensorCombined, VehicleGlobalPosition, ActuatorMotors

class FaultDetection(Node):
    def __init__(self):
        super().__init__('fault_detection')
        
        # Define QoS profile with Best Effort reliability
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # Subscribe to sensor data and motor thrusts with adjusted QoS
        self.subscription_sensor = self.create_subscription(
            SensorCombined,
            '/fmu/out/sensor_combined',
            self.sensor_callback,
            qos_profile)
        
        self.subscription_gps = self.create_subscription(
            VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.gps_callback,
            qos_profile)

        self.subscription_actuator = self.create_subscription(
            ActuatorMotors,
            '/fmu/out/actuator_motors',
            self.actuator_callback,
            qos_profile)

        # Initialize parameters for fault detection
        self.failed_motor = None
        self.motor_threshold = 0.2 # Updated practical threshold for minimum thrust (reduced for realistic in-flight behavior)
        self.thrusts = [0, 0, 0, 0]

        # Flags for failure conditions
        self.roll_failure_motor1 = False
        self.pitch_failure_motor1 = False
        self.roll_failure_motor2 = False
        self.roll_failure_motor2_a = False
        self.pitch_failure_motor2 = False
        self.roll_failure_motor3 = False
        self.roll_failure_motor3_a = False
        self.pitch_failure_motor3 =False
        self.roll_failure_motor4=False
        self.roll_failure_motor4_a=False
        self.pitch_failure_motor4=False
        self.altitude_failure = False
        self.external_failure = False


    def sensor_callback(self, msg):
        # Process sensor data for attitude failures
        gyro_data = msg.gyro_rad
        accel_data = msg.accelerometer_m_s2

        # Check conditions for roll and pitch failures for motor 1
        if (gyro_data[0]) > 3.5:
            self.roll_failure_motor1 = True
        else:
            self.roll_failure_motor1 = False

        if (gyro_data[2]) < -1.5:
            self.pitch_failure_motor1 = True
        else:
            self.pitch_failure_motor1 = False

        # Only publish failure if both roll and pitch conditions are met for motor 1
        if self.roll_failure_motor1 and self.pitch_failure_motor1:
            self.get_logger().error("Failure_Motor_1.")

        # Check conditions for roll and pitch failures for motor 2
        if (gyro_data[0]) < -4:
            self.roll_failure_motor2 = True
        else:
            self.roll_failure_motor2 = False # Check for strong negative roll on the x-axis

        if (gyro_data[1]) > 4:
            self.roll_failure_motor2_a = True
        else:
            self.roll_failure_motor2_a = False # Check for strong positive roll on the y-axis

        if (gyro_data[2]) < -4:
            self.pitch_failure_motor2 = True
        else:
            self.pitch_failure_motor2 = False

        # Only publish failure if both roll and pitch conditions are met (AND logic) for motor 3
        if self.roll_failure_motor2 and self.pitch_failure_motor2 and self.roll_failure_motor2_a: # Comprehensive check for gyro data conditions # Added gyro[2] condition for comprehensive check
            self.get_logger().error("Failure_Motor_2.") # Added a third condition for comprehensive check   


        # Check conditions for roll and pitch failures for motor 3
        if (gyro_data[0]) > 0.2 :
            self.roll_failure_motor3 = True
        else:
            self.roll_failure_motor3 = False # Check for strong negative roll on the x-axis

        if (gyro_data[1]) <-0.4:
            self.roll_failure_motor3_a = True
        else:
            self.roll_failure_motor3_a = False # Check for strong positive roll on the y-axis

        if (gyro_data[2]) > 1:
            self.pitch_failure_motor3 = True
        else:
            self.pitch_failure_motor3 = False

        # Only publish failure if both roll and pitch conditions are met (AND logic) for motor 2
        if self.roll_failure_motor3 and self.pitch_failure_motor3 and self.roll_failure_motor3_a: # Comprehensive check for gyro data conditions # Added gyro[2] condition for comprehensive check
            self.get_logger().error("Failure_Motor_3.") # Added a third condition for comprehensive check  


        # Check conditions for roll and pitch failures for motor 2
        if (gyro_data[0]) > 1.2 :
            self.roll_failure_motor4 = True
        else:
            self.roll_failure_motor4 = False # Check for strong negative roll on the x-axis

        if (gyro_data[1]) >1.5:
            self.roll_failure_motor4_a = True
        else:
            self.roll_failure_motor4_a = False # Check for strong positive roll on the y-axis

        if (gyro_data[2]) > 4.5:
            self.pitch_failure_motor4 = True
        else:
            self.pitch_failure_motor4 = False

        # Only publish failure if both roll and pitch conditions are met (AND logic) for motor 2
        if self.roll_failure_motor4 and self.pitch_failure_motor4 and self.roll_failure_motor4_a: # Comprehensive check for gyro data conditions # Added gyro[2] condition for comprehensive check
            self.get_logger().error("Failure_Motor_4.") # Added a third condition for comprehensive check  

    def gps_callback(self, msg):
        # Process GPS data for altitude failure detection
        alt = msg.alt
        altitude_threshold = 1.0 # Updated practical altitude threshold in meters
        if alt < altitude_threshold:
            if not self.altitude_failure:
                self.altitude_failure = True
                self.get_logger().error("Altitude failure detected.")
        else:
            self.altitude_failure = False

    def actuator_callback(self, msg):
        # Read motor thrust values
        self.thrusts = msg.control[:4] # Assuming the first 4 indices are thrust values
        
        # Check for motor failure by comparing thrust values
        motor_failure_detected = False
        for i, thrust in enumerate(self.thrusts):
            if thrust < self.motor_threshold:
                if self.failed_motor != i:
                    self.failed_motor = i
                    motor_failure_detected = True
                    self.get_logger().error(f'Motor {i+1} failure detected.')
                break
        if not motor_failure_detected:
            self.failed_motor = None

    def external_failure_check(self):
        # Placeholder for external system trigger failure check
        if self.external_failure:
            self.get_logger().error("Failure triggered by external system.")
    
    def check_all_failures(self):
        # This method runs all the failure checks and logs the detected issues
        if self.roll_failure_motor1 or self.pitch_failure_motor1 or self.roll_failure_motor2 or self.pitch_failure_motor2 or self.roll_failure_motor2_a or self.altitude_failure or self.failed_motor is not None:
            self.get_logger().info("Critical failure(s) detected in vehicle components.")
        else:
            self.get_logger().info("All systems operating within normal parameters.")

def main(args=None):
    rclpy.init(args=args)
    fault_detection = FaultDetection()
    rclpy.spin(fault_detection)
    fault_detection.destroy_node()
    rclpy.shutdown()

# Ensure this main function is called if the script is run directly
if __name__ == '__main__':
    main()