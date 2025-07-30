import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import sys

from px4_msgs.msg import ActuatorMotors, VehicleAttitude, VehicleLocalPosition, VehicleAngularVelocity

from rl_controller.PPOAgent import PPOAgent


class RLController(Node):
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

        self.agent = PPOAgent()
        self.episode_reward = 0

        self.previous_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.motor_feedback = np.array([1,1,1,1])

        self.done = False

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
        if (self.done == False):
            # Create actuator msg
            actuator_msg = ActuatorMotors()
            actuator_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            actuator_msg.timestamp_sample = actuator_msg.timestamp

            # Get current state of vehicle
            x0 = np.array([self.vehicle_local_position[0], self.vehicle_local_position[1], self.vehicle_local_position[2],
                                self.vehicle_local_velocity[0], self.vehicle_local_velocity[1], self.vehicle_local_velocity[2],
                                self.vehicle_attitude[0], self.vehicle_attitude[1], self.vehicle_attitude[2], self.vehicle_attitude[3], 
                                self.vehicle_angular_velocity[0], self.vehicle_angular_velocity[1], self.vehicle_angular_velocity[2],
                                self.previous_action[0], self.previous_action[1], self.previous_action[2], self.previous_action[3],
                                self.motor_feedback[0], self.motor_feedback[1], self.motor_feedback[2], self.motor_feedback[3]])

            target_position = np.array([0.0, 0.0, 10.0])  # Target to hover at (0,0,10)
            target_quaternion = np.array([1, 0, 0, 0])
            position_error = np.linalg.norm(x0[0:3] - target_position)
            velocity_error = np.linalg.norm(x0[3:6])  # Target velocity is zero
            attitude_error = np.linalg.norm(x0[6:10] - target_quaternion)        # Target attitude is zero (level)
            angular_velocity_error = np.linalg.norm(x0[10::])  # Target angular velocity is zero


            cost = (4e-3 * position_error +
                        2e-4 * np.linalg.norm(self.previous_action) +
                        1e-4 * attitude_error +
                        3e-4 * angular_velocity_error +
                        5e-4 * velocity_error)

            action, logprob = self.agent.select_action(x0)
            
            reward = -cost
            # Store transition in memory
            self.agent.store_transition(x0, action, logprob, reward, self.done)

            self.episode_reward += reward

            # Apply outputs
            self.actuator_motors[0] = action[0]
            self.actuator_motors[1] = action[1]
            self.actuator_motors[2] = action[2]
            self.actuator_motors[3] = action[3]

            actuator_msg.control = self.actuator_motors

            self.motor_feedback = self.actuator_motors

            # Publish outputs
            self.publisher_actuator_motors.publish(actuator_msg)

            if (self.vehicle_local_position[2] < 0.5 and self.vehicle_local_velocity[2] < -1.0):
                self.done = True

        elif (self.done == True):
            if len(self.agent.memory['logprobs']) > 0:
                self.agent.update()
            else:
                print(f"Episode ended without storing any transitions.")

            self.agent.save_model()
            sys.exit()
        

def main(args=None):
    rclpy.init(args=args)

    rl = RLController()

    rclpy.spin(rl)

    rl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()