from casadi import MX, vertcat, horzcat, diagcat, Function
from acados_template import AcadosModel
import casadi as cs

class NMPC_Model():
    def __init__(self):
        self.name = "Iris"
        
        # Constants
        self.mass = 1.35
        hover_thrust = 0.66678
        self.max_thrust = self.mass * 9.81 / hover_thrust
        self.max_rate = 0.5
        
        
    def get_acados_model(self) -> AcadosModel:
        # Define symbolic variables
        x = MX.sym('x', 13)  # [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        u = MX.sym('u', 4)   # [T1, T2, T3, T4]
        xdot = MX.sym('xdot', 13)  # State derivative
        g = 9.81

        # Extract quaternion components
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]

        # Rotation matrix R (3x3)
        R = vertcat(
            horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
            horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2))
        )

        # Translational dynamics
        p_dot = x[3:6]  # [vx, vy, vz]
        v_dot = (1 / self.mass) * (R @ vertcat(0, 0, u[0])) + vertcat(0, 0, -g)

        # Rotational dynamics
        w = x[10:13]  # Angular velocities [wx, wy, wz]
        q_dot = 0.5 * vertcat(
            -qx * w[0] - qy * w[1] - qz * w[2],
            qw * w[0] + qy * w[2] - qz * w[1],
            qw * w[1] - qx * w[2] + qz * w[0],
            qw * w[2] + qx * w[1] - qy * w[0]
        )

        # Gyroscopic effects and angular velocity dynamics
        def angular_velocity_dynamics(J, J_inv, tau, w):
            # Compute angular momentum (J * w)
            angular_momentum = J @ w  # Matrix-vector multiplication in CasADi
            
            # Compute gyroscopic effects (w × angular_momentum)
            gyroscopic_effects = vertcat(
                w[1] * angular_momentum[2] - w[2] * angular_momentum[1],
                w[2] * angular_momentum[0] - w[0] * angular_momentum[2],
                w[0] * angular_momentum[1] - w[1] * angular_momentum[0]
            )
            
            # Compute the angular velocity dynamics (w_dot)
            tau_minus_gyro = vertcat(*[tau[i] - gyroscopic_effects[i] for i in range(3)])
            w_dot = J_inv @ tau_minus_gyro  # Matrix-vector multiplication in CasADi
            
            return w_dot

        # Inertia matrix for Iris quadrotor (3x3)
        J = diagcat(0.029125, 0.029125, 0.055225)  # Use diagcat to create a diagonal matrix
        J_inv = diagcat(1 / 0.029125, 1 / 0.029125, 1 / 0.055225)  # Inverse of diagonal matrix
        
       # Define tau as a CasADi symbolic variable (a 4-dimensional vector for torques [tau_x, tau_y, tau_z])
        tau = cs.MX.sym('tau', 4)

        # Define G as a CasADi matrix (4x4)
        G = cs.MX(4, 4)  # 4x4 matrix

        # Define the rows of G
        G[0, :] = [1, 1, 1, 1]  # First row
        G[1, :] = [0.22, -0.2, -0.22, 0.2]  # Second row
        G[2, :] = [-0.13, 0.13, -0.13, 0.13]  # Third row
        G[3, :] = [-0.06, -0.06, 0.06, 0.06]  # Fourth row
        
        tau = G @ u
        tau=tau[1:4]
        
        # Compute w_dot using angular velocity dynamics
        w_dot = angular_velocity_dynamics(J, J_inv, tau, w)

        # Concatenate full dynamics
        x_dot = vertcat(p_dot, v_dot, q_dot, w_dot)

        # Create and configure the Acados model
        model = AcadosModel()
        model.x = x
        model.xdot = xdot  # Explicitly define state derivatives
        model.u = u
        model.f_expl_expr = x_dot  # Define dynamics as f(x, u)
        model.name = self.name

        return model