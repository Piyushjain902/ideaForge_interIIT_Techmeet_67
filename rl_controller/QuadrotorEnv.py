import numpy as np
import torch
import random



# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Environment Setup ---
class QuadrotorEnv:
    def __init__(self, max_steps=200, dt=0.01):
        """
        Initializes the Quadrotor environment.

        Args:
            max_steps (int): Maximum number of steps per episode.
            dt (float): Time step duration.
        """
        self.max_steps = max_steps
        self.dt = dt  # Time step
        self.reset()

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
            np.ndarray: The initial state vector.
        """
        # Initialize state components
        self.position = np.random.uniform(low=-1.0, high=1.0, size=3)
        self.linear_velocity = np.random.uniform(low=-1.0, high=1.0, size=3)
        self.attitude = self.random_attitude()
        self.angular_velocity = np.random.uniform(low=-1.0, high=1.0, size=3)
        self.previous_action = np.zeros(4)
        self.motor_feedback = self.initialize_motor_feedback()
        self.current_step = 0
        self.done = False

        # Randomly decide the number of working motors: 4, 3, or 2
        self.num_working_motors = np.random.choice([4, 3, 2])
        self.working_motors = np.random.choice(range(4), self.num_working_motors, replace=False)

        return self.get_state()

    def random_attitude(self):
        """
        Generates random Euler angles between -pi and pi.

        Returns:
            np.ndarray: Attitude vector (pitch, roll, yaw).
        """
        return np.random.uniform(low=-np.pi, high=np.pi, size=3)

    def initialize_motor_feedback(self):
        """
        Initializes motor feedback indicating operational status.

        Returns:
            np.ndarray: Motor feedback vector.
        """
        feedback = np.zeros(4)
        for motor in self.working_motors:
            feedback[motor] = 1.0  # 1.0 indicates working motor
        return feedback

    def step(self, action):
        """
        Applies an action to the environment and updates the state.

        Args:
            action (np.ndarray): 4-dimensional action vector representing angular velocities.

        Returns:
            tuple:
                np.ndarray: Next state vector.
                float: Reward.
                bool: Whether the episode has ended.
                dict: Additional info (empty in this case).
        """
        # Clip action to ensure it's within valid range
        angular_velocity_command = np.clip(action, -1.0, 1.0)

        # Set angular velocities to zero for failed motors
        for i in range(4):
            if i not in self.working_motors:
                angular_velocity_command[i] = 0.0

        # Update angular velocities
        self.angular_velocity += angular_velocity_command * self.dt

        # Update attitude based on angular velocity
        self.attitude += self.angular_velocity * self.dt
        self.attitude = self.wrap_angles(self.attitude)

        # Update linear acceleration based on motor feedback and current attitude
        # Simplified physics: thrust affects vertical acceleration
        # Assume each working motor provides upward thrust proportional to its feedback
        total_thrust = np.sum(self.motor_feedback[self.working_motors])
        gravity = 9.81
        self.linear_acceleration = np.array([0.0, 0.0, (total_thrust - gravity) / 1.0])  # mass = 1 kg

        # Update linear velocities
        self.linear_velocity += self.linear_acceleration * self.dt

        # Update positions
        self.position += self.linear_velocity * self.dt

        # Update previous action
        self.previous_action = angular_velocity_command.copy()

        # Compute reward (negative cost)
        target_position = np.array([0.0, 0.0, 10.0])  # Target to hover at (0,0,10)
        position_error = np.linalg.norm(self.position - target_position)
        velocity_error = np.linalg.norm(self.linear_velocity)  # Target velocity is zero
        attitude_error = np.linalg.norm(self.attitude)        # Target attitude is zero (level)
        angular_velocity_error = np.linalg.norm(self.angular_velocity)  # Target angular velocity is zero

        # Define thresholds for terminal states
        position_threshold = 0.1
        velocity_threshold = 0.1
        attitude_threshold = 0.1
        angular_velocity_threshold = 0.1

        # Check termination conditions
        self.current_step += 1
        if (position_error < position_threshold and velocity_error < velocity_threshold and
            attitude_error < attitude_threshold and angular_velocity_error < angular_velocity_threshold):
            self.done = True
        if self.current_step >= self.max_steps:
            self.done = True
        if self.position[2] <= 0:
            self.done = True  # Failure: quadrotor touched the ground

        # Reward shaping
        if self.num_working_motors == 4:
            # Include yaw angle in reward calculation
            yaw_error = abs(self.attitude[2])  # Assuming attitude[2] is yaw
            cost = (4e-3 * position_error +
                    2e-4 * np.linalg.norm(angular_velocity_command) +
                    3e-4 * angular_velocity_error +
                    5e-4 * velocity_error +
                    1e-3 * yaw_error)
        else:
            # Exclude yaw angle in reward calculation
            cost = (4e-3 * position_error +
                    2e-4 * np.linalg.norm(angular_velocity_command) +
                    3e-4 * angular_velocity_error +
                    5e-4 * velocity_error)

        reward = -cost  # Negative cost as reward

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        """
        Concatenates all state components into a single vector.

        Returns:
            np.ndarray: State vector.
        """
        state = np.concatenate([
            self.position,                # 3
            self.linear_velocity,         # 3
            self.attitude,                # 3
            self.angular_velocity,        # 3
            self.previous_action,         # 4
            self.motor_feedback           # 4
        ])  # Total: 20
        return state.copy()

    @staticmethod
    def wrap_angles(angles):
        """
        Wraps angles to be within [-pi, pi].

        Args:
            angles (np.ndarray): Array of angles.

        Returns:
            np.ndarray: Wrapped angles.
        """
        return (angles + np.pi) % (2 * np.pi) - np.pi
