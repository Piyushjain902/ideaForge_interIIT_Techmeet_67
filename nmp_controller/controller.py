import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, ocp_get_default_cmake_builder

class NMPC_Controller:
    def __init__(self, model,x0):
        self.model = model
        self.Tf = 1.0  # Prediction horizon (in seconds)
        self.N = 20    # Number of prediction steps
        
        self.x0=x0
        # Initialize solver and integrator
        self.ocp_solver, self.integrator = self.setup(self.x0,self.N, self.Tf)

    def setup(self, x0,N_horizon, Tf):
        ocp = AcadosOcp()

        # Set the system model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust
        wmax = self.model.max_rate
        mass=self.model.mass
        g=9.81
        wei=mass*g

        ocp.model = model

        nx = model.x.size()[0]  # Number of states (13)
        nu = model.u.size()[0]  # Number of controls (4)

        # Set dimensions
        ocp.dims.N = N_horizon

        # Define weights for the cost function
        W = np.diag([80, 80, 800,    # Position weights (p)
                    60,             # Quaternion-derived term r
                    0,              # Quaternion-derived term q_z_z
                    1, 1, 1,        # Velocity weights (v)
                    0.5, 0.5, 0.1,  # Angular velocity weights (omega)
                    1, 1, 1, 1])    # Control input weights (u)
        
        W_terminal = np.diag([80, 80, 800,    # Position weights (p)
                             1, 1, 1])             # Quaternion-derived term r
                            
        # Assign weights to the cost function
        ocp.cost.W = W
        ocp.cost.W_e = W_terminal
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'

        # Extract state and control variables
        p = model.x[0:3]   # Position (x, y, z)
        v = model.x[3:6]   # Velocity (vx, vy, vz)
        q = model.x[6:10]  # Quaternion (qw, qx, qy, qz)
        omega = model.x[10:13]  # Angular velocity (wx, wy, wz)
        u = model.u  # Control inputs (F, tau_x, tau_y, tau_z)

        # Reference values (should be symbolic or np arrays)
        model.p_ref = cs.MX([0, 0, 15])  # Target position
        model.v_ref = cs.MX([0, 0, 0])   # Target velocity (symbolic variable)
        model.omega_ref = cs.MX([0, 0, 0])  # Target angular velocity (symbolic variable)
        model.u_ref = cs.MX([wei/4, wei/4, wei/4, wei/4])  # Target control inputs

        # Compute quaternion error q_e = q_ref * q_inv
        q_ref = cs.vertcat(1, 0, 0, 0)  # Target quaternion (neutral orientation)
        q_inv = cs.vertcat(q[0], -q[1], -q[2], -q[3])  # Quaternion inverse

        q_e = cs.vertcat(
            q_ref[0] * q_inv[0] - q_ref[1] * q_inv[1] - q_ref[2] * q_inv[2] - q_ref[3] * q_inv[3],
            q_ref[0] * q_inv[1] + q_ref[1] * q_inv[0] + q_ref[2] * q_inv[3] - q_ref[3] * q_inv[2],
            q_ref[0] * q_inv[2] - q_ref[1] * q_inv[3] + q_ref[2] * q_inv[0] + q_ref[3] * q_inv[1],
            q_ref[0] * q_inv[3] + q_ref[1] * q_inv[2] - q_ref[2] * q_inv[1] + q_ref[3] * q_inv[0]
        )

        # Quaternion error components
        p_dash = (q_e[1] ** 2 + q_e[2] ** 2) / (q_e[0] ** 2 + q_e[3] ** 2)  # Example calculation
        r = p_dash / (1 - p_dash)
        q_z_z = q_e[3] / cs.sqrt(1 + r)  # Simplified quaternion term

        # Define y_i as per your request
        y_i = cs.vertcat(
            p - model.p_ref,  # Position error
            r,               # Quaternion x and y components squared
            q_z_z,           # Simplified quaternion z component
            v - model.v_ref,  # Velocity error
            omega - model.omega_ref,  # Angular velocity error
            u - model.u_ref   # Control input error
        )

        # Define terminal cost
        y_e = cs.vertcat(p, v)  # Terminal cost only includes position and velocity

        # Set cost terms in OCP
        ocp.cost.yref = np.zeros((y_i.shape[0],))  # Reference for the cost
        ocp.cost.yref_e = np.zeros((y_e.shape[0],))  # Terminal reference
        ocp.model.cost_y_expr = y_i
        ocp.model.cost_y_expr_e = y_e

        # Define constraints for control inputs
        ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0])  # Lower bounds for inputs
        ocp.constraints.ubu = np.array([5.25, 5.25, 5.25, 5.25])    # Upper bounds for inputs
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])              # Input indices for constraints
        ocp.constraints.x0=x0

        # Solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.tf = Tf
        
        # Create the solver and integrator
        ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        acados_integrator = AcadosSimSolver(ocp, json_file='acados_ocp.json')
        
        builder = ocp_get_default_cmake_builder()
        ocp_solver.generate(ocp, cmake_builder=builder, json_file='acados_ocp.json')

        return ocp_solver, acados_integrator
    
    def solve(self, x0):
            """
            Solves the NMPC problem by setting the initial state, solving for the optimal control sequence,
            and then retrieving the control inputs and state trajectory.

            Args:
            - x0: Initial state of the system.

            Returns:
            - simU: Control inputs over the prediction horizon.
            - simX: State trajectory over the prediction horizon.
            """
            
            # Set the initial state in the solver
            self.ocp_solver.set(0, "lbx", x0)  # Lower bounds for state
            self.ocp_solver.set(0, "ubx", x0)  # Upper bounds for state

            # Solve the NMPC problem
            status = self.ocp_solver.solve()
            if status != 0:
                raise Exception(f"acados returned status {status}. The solver failed!")

            # Retrieve the solution
            nx = self.model.get_acados_model().x.size()[0]  # Number of states (13)
            nu = self.model.get_acados_model().u.size()[0]  # Number of controls (4)
            N = self.N  # Prediction horizon length

            # Initialize arrays to store the state trajectory and control inputs
            simX = np.zeros((N + 1, nx))  # State trajectory
            simU = np.zeros((N, nu))      # Control inputs

            # Extract the state and control inputs over the prediction horizon
            for i in range(N):
                simX[i, :] = self.ocp_solver.get(i, "x")  # Get state at time step i
                simU[i, :] = self.ocp_solver.get(i, "u")  # Get control input at time step i

            # Retrieve the terminal state at the last step
            simX[N, :] = self.ocp_solver.get(N, "x")

            # The final result will be the control inputs (simU) and the state trajectory (simX)
            return simU, simX

