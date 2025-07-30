import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import ActuatorMotors, VehicleAttitude, VehicleLocalPosition, VehicleAngularVelocity

from nmp_controller.model import NMPC_Model
from nmp_controller.controller import NMPC_Controller

class NMPC(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscriber for orientation
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile)
        
        # Create subscriber for position and velocity
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile)
        
        # Create subscriber for angular velocity
        self.angular_velocity_sub = self.create_subscription(
            VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.vehicle_angular_velocity_callback,
            qos_profile)

        # Create publisher to publish commands
        self.publisher_actuator_motors = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.actuator_motors = np.array(dtype=np.float32, object=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0])
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0])
        self.vehicle_angular_velocity = np.array([0.0, 0.0, 0.0])

        # NMPC model for drone
        self.drone_model = NMPC_Model()

        # Main controller
        self.drone_controller = NMPC_Controller(self.drone_model, np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))

    # Callback to update orientation
    def vehicle_attitude_callback(self, msg):
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    # Callback to update angular velocity
    def vehicle_angular_velocity_callback(self, msg):
        self.vehicle_angular_velocity[0] = msg.xyz[0]
        self.vehicle_angular_velocity[1] = -msg.xyz[1]
        self.vehicle_angular_velocity[2] = -msg.xyz[2]

    # Callback to update position and velocity
    def vehicle_local_position_callback(self, msg):
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    # Main loop
    def cmdloop_callback(self):
        # Create actuator msg
        actuator_msg = ActuatorMotors()
        actuator_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        actuator_msg.timestamp_sample = actuator_msg.timestamp

        # Get current state of vehicle
        x0 = np.array([self.vehicle_local_position[0], self.vehicle_local_position[1], self.vehicle_local_position[2],
                            self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
                            self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3], 
                            self.vehicle_angular_velocity[0], self.vehicle_angular_velocity[1], self.vehicle_angular_velocity[2]])
        
        # Solve for actuator outputs
        u, x = self.drone_controller.solve(x0)

        # Apply outputs
        self.actuator_motors[0] = u[0][0]
        self.actuator_motors[1] = u[0][1]
        self.actuator_motors[2] = u[0][2]
        self.actuator_motors[3] = u[0][3]
        actuator_msg.control = self.actuator_motors

        # Publish outputs
        self.publisher_actuator_motors.publish(actuator_msg)
        

def main(args=None):
    rclpy.init(args=args)

    nmpc = NMPC()

    rclpy.spin(nmpc)

    nmpc.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
