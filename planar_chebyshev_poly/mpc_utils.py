import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- User-Defined Parameters ---
n_x = 6 # Number of states (px, pz, phi, vx, vz, phi_dot)
n_u = 2 # Number of controls (u1, u2 - thrust forces from two propellers)

# VTOL physical parameters (from the document)
m = 0.486  # Mass [kg]
J = 0.00383  # Moment of inertia [kg⋅m²]
l = 0.25   # Distance between propellers and center of mass [m]
g = 9.81   # Gravity [m/s²]

# load optimization results
data = np.load('vtol_optimization_results.npz')
X_opt = data['X_opt']
U_opt = data['U_opt']
t_opt = data['t_opt']
T_opt = t_opt[-1]

# Simulation parameters
t_sim = T_opt
dt_sim = 0.005  # Simulation time step
t_sim = np.arange(0, t_sim, dt_sim)
N_sim = len(t_sim)

# --- Variable Bounds and Scaling ---
# States: [px, pz, phi, vx, vz, phi_dot]
x_min = np.array([-3, 0, -np.pi, -7.5, -7.5, -15])  # Lower bounds
x_max = np.array([3, 3, np.pi, 7.5, 7.5, 15])         # Upper bounds
x_scale = x_max - x_min  # Scaling factor for each state
x_shift = x_min          # Shift for each state

# Controls: [u1, u2] (thrust forces)
u_min = np.array([0.0, 0.0])
u_max = np.array([5.0, 5.0])
u_scale = u_max - u_min
u_shift = u_min

# --- Scaling and Descaling Functions ---
def scale_casadi(var_phys, var_shift, var_scale):
    """Scale physical state to normalized [0, 1] - CasADi version"""
    x_norm = ca.MX.zeros(var_phys.shape)
    for i in range(var_phys.shape[0]):
        x_norm[i, :] = (var_phys[i, :] - var_shift[i]) / var_scale[i]
    return x_norm

def descale_casadi(var_norm, var_shift, var_scale):
    """Descale normalized state [0, 1] to physical units - CasADi version"""
    x_phys = ca.MX.zeros(var_norm.shape)
    for i in range(var_norm.shape[0]):
        x_phys[i, :] = var_norm[i, :] * var_scale[i] + var_shift[i]
    return x_phys

# --- Collocation Node Options ---
COLLOCATION_TYPE = 'chebyshev'  # Options: 'chebyshev' or 'legendre'

# --- 1. Local Chebyshev-Gauss-Lobatto (CGL) Nodes per interval ---
def cgl_nodes(N):
    """Compute Chebyshev-Gauss-Lobatto nodes in [-1, 1]"""
    return np.flip(np.cos(np.pi * np.arange(N + 1) / N))

def lgl_nodes(N):
    """Compute Legendre-Gauss-Lobatto nodes in [-1, 1]"""
    if N == 0:
        return np.array([0.0])
    elif N == 1:
        return np.array([-1.0, 1.0])
    
    # Use scipy for LGL nodes
    from scipy.special import legendre
    from scipy.optimize import fsolve
    
    # LGL nodes: -1, 1, and roots of P'_N(x)
    x = np.zeros(N + 1)
    x[0] = -1.0
    x[N] = 1.0
    
    if N > 1:
        # Derivative of Legendre polynomial
        P_N = legendre(N)
        dP_N = np.polyder(P_N)
        
        # Initial guess using Chebyshev nodes
        x_init = np.cos(np.pi * np.arange(1, N) / N)
        
        # Solve for interior roots
        x[1:N] = fsolve(lambda xi: np.polyval(dP_N, xi), x_init)
    
    return np.sort(x)[::-1]  # Return in descending order like CGL

def get_collocation_nodes(N, method='chebyshev'):
    """Get collocation nodes based on method"""
    if method == 'chebyshev':
        return cgl_nodes(N)
    elif method == 'legendre':
        return lgl_nodes(N)
    else:
        raise ValueError(f"Unknown collocation method: {method}")

# --- Chebyshev Basis Functions ---
def chebyshev_polynomial(n, x):
    """Compute the nth Chebyshev polynomial T_n(x) using recurrence relation"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        T_prev_prev = np.ones_like(x)  # T_0
        T_prev = x                     # T_1
        for _ in range(2, n + 1):
            T_curr = 2 * x * T_prev - T_prev_prev
            T_prev_prev = T_prev
            T_prev = T_curr
        return T_prev

def chebyshev_basis_matrix(N, tau):
    """Build matrix of Chebyshev polynomials evaluated at CGL nodes"""
    T_matrix = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            T_matrix[i, j] = chebyshev_polynomial(j, tau[i])
    return T_matrix

# --- Legendre Basis Functions ---
def legendre_polynomial(n, x):
    """Compute the nth Legendre polynomial P_n(x) using recurrence relation"""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    else:
        P_prev_prev = np.ones_like(x)  # P_0
        P_prev = x                      # P_1
        for k in range(2, n + 1):
            P_curr = ((2*k - 1) * x * P_prev - (k - 1) * P_prev_prev) / k
            P_prev_prev = P_prev
            P_prev = P_curr
        return P_prev

# --- Differentiation Matrix ---
def chebyshev_diff_matrix(N, tau):
    """Differentiation matrix for Chebyshev-Gauss-Lobatto nodes"""
    D = np.zeros((N + 1, N + 1))
    
    # Weights for the differentiation matrix
    w = np.ones(N + 1)
    w[0] = w[N] = 0.5
    
    # Differentiation matrix formula for CGL nodes (ascending order [-1, +1])
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = w[j] / w[i] * (-1)**(i + j) / (tau[i] - tau[j])
            elif i == j and i != 0 and i != N:
                D[i, j] = -tau[i] / (2 * (1 - tau[i]**2))
            elif i == 0:  # tau = -1 (start)
                D[0, 0] = -(2*N**2 + 1)/6
            elif i == N:  # tau = +1 (end)
                D[N, N] = (2*N**2 + 1)/6
    
    return D

def legendre_diff_matrix(N, tau):
    """Differentiation matrix for Legendre-Gauss-Lobatto nodes"""
    from scipy.special import legendre
    
    D = np.zeros((N + 1, N + 1))
    P_N = legendre(N)
    
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                P_i = np.polyval(P_N, tau[i])
                P_j = np.polyval(P_N, tau[j])
                D[i, j] = P_i / (P_j * (tau[i] - tau[j]))
            else:
                D[i, i] = 0.0  # Diagonal elements
    
    # Boundary corrections for LGL
    D[0, 0] = -N * (N + 1) / 4.0
    D[N, N] = N * (N + 1) / 4.0
    
    return D

def get_diff_matrix(N, tau, method='chebyshev'):
    """Get differentiation matrix based on method"""
    if method == 'chebyshev':
        return chebyshev_diff_matrix(N, tau)
    elif method == 'legendre':
        return legendre_diff_matrix(N, tau)
    else:
        raise ValueError(f"Unknown differentiation method: {method}")

# --- Planar VTOL Vehicle Dynamics ---
# --- Planar VTOL Vehicle Dynamics ---
# Create symbolic variables for CasADi Function
x_sym = ca.SX.sym('x', 6)
u_sym = ca.SX.sym('u', 2)

# Define dynamics symbolically
phi, vx, vz, phi_dot = x_sym[2], x_sym[3], x_sym[4], x_sym[5]
u1, u2 = u_sym[0], u_sym[1]

cos_phi = ca.cos(phi)
sin_phi = ca.sin(phi)

px_dot = vx
pz_dot = vz
phi_dot_out = phi_dot
vx_dot = -(u1 + u2) * sin_phi / m
vz_dot = (u1 + u2) * cos_phi / m - g
phi_ddot = l * (u1 - u2) / J

f_dynamics_expr = ca.vertcat(px_dot, pz_dot, phi_dot_out, vx_dot, vz_dot, phi_ddot)

# Create CasADi Functions
# f_dynamics: For use in optimization (symbolic)
# f_dynamics_casadi: For numerical evaluation in simulation
f_dynamics = ca.Function('f_dynamics', [x_sym, u_sym], [f_dynamics_expr])
f_dynamics_casadi = f_dynamics  # Same function, can be used for both symbolic and numerical

# Create CasADi Function for RK4 integration
dt_rk4 = ca.SX.sym('dt_rk4')

k1 = f_dynamics(x_sym, u_sym)
k2 = f_dynamics(x_sym + dt_rk4/2 * k1, u_sym)
k3 = f_dynamics(x_sym + dt_rk4/2 * k2, u_sym)
k4 = f_dynamics(x_sym + dt_rk4 * k3, u_sym)
x_next = x_sym + dt_rk4/6 * (k1 + 2*k2 + 2*k3 + k4)

rk4_step_casadi = ca.Function('rk4_step', [x_sym, u_sym, dt_rk4], [x_next])

def rk4_step(x, u, dt):
    """RK4 integration step for VTOL dynamics using CasADi Function
    Args:
        x: Current state vector (numpy array)
        u: Current control vector (numpy array)
        dt: Time step
    Returns:
        Next state vector (numpy array)
    """
    # Use the CasADi Function and convert result to numpy array
    result = rk4_step_casadi(x, u, dt)
    return np.array(result).flatten()

# --- MPC Path Tracking with Pseudospectral Collocation ---
def create_mpc_controller(T_horizon, N_intervals, N_per, use_scaling=True, collocation_method='chebyshev'):
    """
    Create MPC controller with pseudospectral collocation (Chebyshev or Legendre)
    
    Args:
        T_horizon: Prediction horizon [s]
        N_intervals: Number of collocation intervals
        N_per: Nodes per interval
        use_scaling: If True, use normalized variables [0,1]; if False, use physical variables directly
        collocation_method: 'chebyshev' for CGL nodes or 'legendre' for LGL nodes
    """
    print(f"   Creating MPC with {collocation_method.upper()} pseudospectral collocation")
    # Cost weights in physical space (what we care about)
    Q_physical = ca.diag([100.0, 100.0, 50.0, 1.0, 1.0, 5.0])  # State tracking weights [px, pz, phi, vx, vz, phi_dot]
    R_physical = ca.diag([10.0, 10.0])                           # Control effort weights (higher to penalize large controls)
    Q_terminal_physical = ca.diag([50.0, 50.0, 25.0, 5.0, 5.0, 2.5])  # Terminal cost weights (high to reach target)
    
    # Scale cost weights for normalized variables
    # Cost in physical: (x_phys - x_ref)^T Q_phys (x_phys - x_ref)
    # Cost in normalized: (x_norm - x_ref_norm)^T Q_scaled (x_norm - x_ref_norm)
    # where x_phys = x_norm * scale, so Q_scaled = Q_phys * scale^2
    if use_scaling:
        Q_diag = [Q_physical[i,i] * x_scale[i]**2 for i in range(n_x)]
        R_diag = [R_physical[i,i] * u_scale[i]**2 for i in range(n_u)]
        Q_terminal_diag = [Q_terminal_physical[i,i] * x_scale[i]**2 for i in range(n_x)]
    else:
        Q_diag = [Q_physical[i,i] for i in range(n_x)]
        R_diag = [R_physical[i,i] for i in range(n_u)]
        Q_terminal_diag = [Q_terminal_physical[i,i] for i in range(n_x)]
    
    Q = ca.diag(ca.DM(Q_diag))
    R = ca.diag(ca.DM(R_diag))
    Q_terminal = ca.diag(ca.DM(Q_terminal_diag))

    # MPC Parameters (passed as function arguments)
    N_mpc = N_intervals * N_per  # Total number of nodes in MPC horizon
    dt_mpc = T_horizon / N_intervals  # Time step per interval

    # MPC Optimization Problem
    opti_mpc = ca.Opti()

    if use_scaling:
        # Normalized decision variables in [0, 1]
        X_norm = opti_mpc.variable(n_x, N_mpc + 1)  # Normalized states
        U_norm = opti_mpc.variable(n_u, N_mpc + 1)  # Normalized controls
        
        # Physical variables using descaling functions
        X_mpc = descale_casadi(X_norm, x_shift, x_scale)
        U_mpc = descale_casadi(U_norm, u_shift, u_scale)
        
        # Bounds on normalized variables [0, 1]
        for i in range(n_x):
            opti_mpc.subject_to(opti_mpc.bounded(0, X_norm[i, :], 1))
        for i in range(n_u):
            opti_mpc.subject_to(opti_mpc.bounded(0, U_norm[i, :], 1))
    else:
        # Physical decision variables directly (no scaling)
        X_mpc = opti_mpc.variable(n_x, N_mpc + 1)
        U_mpc = opti_mpc.variable(n_u, N_mpc + 1)
        X_norm = X_mpc  # For compatibility, point to same variables
        U_norm = U_mpc
        
        # Physical bounds directly - applied to ALL nodes across ALL intervals
        for i in range(n_x):
            opti_mpc.subject_to(opti_mpc.bounded(x_min[i], X_mpc[i, :], x_max[i]))
        for i in range(n_u):
            opti_mpc.subject_to(opti_mpc.bounded(u_min[i], U_mpc[i, :], u_max[i]))
    # Initialize total cost accumulator
    total_tracking_cost = 0
    
    # Parameters (set at each MPC iteration) - in physical units
    x_current = opti_mpc.parameter(n_x)      # Current state
    x_ref_traj = opti_mpc.parameter(n_x, N_mpc + 1)  # Reference trajectory
    u_ref_traj = opti_mpc.parameter(n_u, N_mpc + 1)  # Reference controls

    # Initial condition (in physical space)
    opti_mpc.subject_to(X_mpc[:, 0] == x_current)

    # Build global CGL nodes for the entire horizon with shared boundaries
    # Each interval uses descending tau [-1, +1], but we map to ascending time
    # Global structure: [t0, ..., t_T] with N_intervals * N_per + 1 nodes
    # Interval i spans global indices [i*N_per, (i+1)*N_per]
    
    for i in range(N_intervals):
        # Get collocation nodes and differentiation matrix based on method
        tau_loc = get_collocation_nodes(N_per, collocation_method)
        D_loc = get_diff_matrix(N_per, tau_loc, collocation_method)
        D_loc_ca = ca.DM(D_loc)
        
        # Extract states for this interval (ascending in time)
        X_interval = X_mpc[:, i*N_per:(i+1)*N_per+1]  # shape: (n_x, N_per+1) - physical
        U_interval = U_mpc[:, i*N_per:(i+1)*N_per+1]  # shape: (n_u, N_per+1) - physical
        X_ref_interval = x_ref_traj[:, i*N_per:(i+1)*N_per+1]  # physical reference
        U_ref_interval = u_ref_traj[:, i*N_per:(i+1)*N_per+1]  # physical reference
        
        # Prepare variables for cost computation (scaled or unscaled)
        if use_scaling:
            # Also extract normalized variables for cost computation
            X_norm_interval = X_norm[:, i*N_per:(i+1)*N_per+1]
            U_norm_interval = U_norm[:, i*N_per:(i+1)*N_per+1]
            
            # Normalize reference trajectory for cost computation using scaling functions
            X_ref_norm_interval = scale_casadi(X_ref_interval, x_shift, x_scale)
            U_ref_norm_interval = scale_casadi(U_ref_interval, u_shift, u_scale)
        else:
            # Use physical variables directly for cost
            X_norm_interval = X_interval
            U_norm_interval = U_interval
            X_ref_norm_interval = X_ref_interval
            U_ref_norm_interval = U_ref_interval

        # CGL nodes now in ascending order [-1, ..., +1] matching time order
        # No reversal needed - use directly
        X_interval_cgl = X_interval  # Physical, ordered as tau: [-1, ..., +1]
        U_interval_cgl = U_interval  # Physical
        X_norm_interval_cgl = X_norm_interval  # For cost computation
        U_norm_interval_cgl = U_norm_interval  # For cost computation
        X_ref_norm_interval_cgl = X_ref_norm_interval  # Reference for cost
        U_ref_norm_interval_cgl = U_ref_norm_interval  # Reference for cost
        
        u_rate_max = 2 # Maximum control rate [N/s]
        
        # Apply collocation constraints at each CGL node
        for j in range(N_per + 1):
            # Control rate constraints using Chebyshev differentiation
            # du/dtau = D * u, then du/dt = du/dtau * dtau/dt = D * u * (2/dt_mpc)
            du1_dt = ca.mtimes(D_loc_ca[j, :], U_interval_cgl[0, :].T) * (2/dt_mpc)
            du2_dt = ca.mtimes(D_loc_ca[j, :], U_interval_cgl[1, :].T) * (2/dt_mpc)

            opti_mpc.subject_to(opti_mpc.bounded(-u_rate_max, du1_dt, u_rate_max))
            opti_mpc.subject_to(opti_mpc.bounded(-u_rate_max, du2_dt, u_rate_max))
            
            # State derivative using CGL differentiation matrix (use physical variables for dynamics)
            state_derivative = ca.mtimes(D_loc_ca[j, :], X_interval_cgl.T) * (2/dt_mpc)
            dynamics_rhs = f_dynamics(X_interval_cgl[:, j], U_interval_cgl[:, j])
            opti_mpc.subject_to(state_derivative.T == dynamics_rhs)
            
            # Accumulate tracking cost in NORMALIZED space (properly scaled)
            state_error_norm = X_norm_interval_cgl[:, j] - X_ref_norm_interval_cgl[:, j]
            control_error_norm = U_norm_interval_cgl[:, j] - U_ref_norm_interval_cgl[:, j]
            stage_cost = ca.mtimes([state_error_norm.T, Q, state_error_norm]) + ca.mtimes([control_error_norm.T, R, control_error_norm])
            total_tracking_cost += stage_cost  # Accumulate INSIDE the loop!
            
    # Continuity between intervals is automatic with shared global indexing
    
    # Terminal cost - heavily penalize final state deviation
    if use_scaling:
        # Normalize final reference state using scaling function
        x_ref_final = ca.reshape(x_ref_traj[:, -1], (n_x, 1))
        x_ref_final_norm = scale_casadi(x_ref_final, x_shift, x_scale)
        x_terminal_error = X_norm[:, -1] - x_ref_final_norm[:, 0]
    else:
        # Use physical variables directly
        x_terminal_error = X_mpc[:, -1] - x_ref_traj[:, -1]
    
    terminal_cost = ca.mtimes([x_terminal_error.T, Q_terminal, x_terminal_error])
    
    # Single minimize with sum of all costs
    opti_mpc.minimize(total_tracking_cost + terminal_cost)
    
    # Configure solver
    if use_scaling:
        # With normalized variables, manual scaling is sufficient
        opti_mpc.solver('ipopt', {'print_time': False}, {
            'tol': 1e-3,  # Relaxed for faster convergence
            'acceptable_tol': 1e-2,  # Allow acceptable solutions
            'max_iter': 200,
            'linear_solver': 'mumps',
            'print_level': 0,
            'nlp_scaling_method': 'none',  # Disable auto-scaling since we manually scaled to [0,1]
            'warm_start_init_point': 'yes',  # Use warm start
            'mu_strategy': 'adaptive',  # Adaptive barrier parameter
        })
    else:
        # Without scaling, let IPOPT handle gradient-based scaling
        opti_mpc.solver('ipopt', {'print_time': False}, {
            'tol': 1e-3,
            'max_iter': 200,
            'linear_solver': 'mumps',
            'print_level': 0,
            'nlp_scaling_method': 'gradient-based',  # Let IPOPT scale automatically
            'warm_start_init_point': 'yes',  # Use warm start
            'mu_strategy': 'adaptive',  # Adaptive barrier parameter
        })

    # Extract symbolic expressions
    g = opti_mpc.g  # constraints
    x_all = opti_mpc.x  # decision variables
    Jg = ca.jacobian(g, x_all)
    plt.figure()
    plt.spy(Jg.sparsity())
    plt.title("Jacobian of constraints (Opti)")
    plt.show()
  
    return opti_mpc, x_current, x_ref_traj, u_ref_traj, X_mpc, U_mpc

# --- Reference Path Generation ---
def generate_reference_path(t_sim):
    """Generate a reference path for the VTOL to follow using the optimal trajectory"""
    # Use the previously calculated optimal trajectory as reference
    
    # Interpolate the optimal trajectory to match simulation time vector
    x_ref_full = np.zeros((n_x, len(t_sim)))
    u_ref_full = np.zeros((n_u, len(t_sim)))
    
    for state_idx in range(n_x):
        x_ref_full[state_idx, :] = np.interp(t_sim, t_opt, X_opt[state_idx, :])
    
    for control_idx in range(n_u):
        u_ref_full[control_idx, :] = np.interp(t_sim, t_opt, U_opt[control_idx, :])
    
    return x_ref_full, u_ref_full