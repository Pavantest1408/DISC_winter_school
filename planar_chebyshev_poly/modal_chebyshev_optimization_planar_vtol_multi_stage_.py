#%%
import casadi as ca
import numpy as np
from matplotlib.animation import FuncAnimation
import time


# --- User-Defined Parameters ---
# Multiple-interval Chebyshev setup
S_intervals = 300          # number of sub-intervals (stages)
N_per = 6               # Chebyshev nodes per interval (CGL, N_per+1 nodes)
n_x = 6 # Number of states (px, pz, phi, vx, vz, phi_dot)
n_u = 2 # Number of controls (u1, u2 - thrust forces from two propellers)

# --- Basis Function Selection ---
USE_CHEBYSHEV_BASIS = False  # Use Chebyshev polynomials with coefficient constraints
COEFFICIENT_CONSTRAINTS = True  # Embed coefficient solving as constraints

# --- Smoothing Weights ---
# Penalize control rate to discourUSE_CHEBYSHEV_BASISage sharp jumps within intervals
LAMBDA_U_RATE = 1e0  # weight for integral of ||du/dt||^2

# VTOL physical parameters (from the document)
m = 0.486  # Mass [kg]
J = 0.00383  # Moment of inertia [kg⋅m²]
l = 0.25   # Distance between propellers and center of mass [m]
g = 9.81   # Gravity [m/s²]

# --- 1. Local Chebyshev-Gauss-Lobatto (CGL) Nodes per interval ---
# Standard CGL: tau_j = cos(pi*j/N_per), j=0..N_per
j_loc = np.arange(0, N_per + 1)
tau_loc = np.cos(np.pi * j_loc / N_per)

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

# Compute local basis matrix for convenience (if needed)
T_basis_loc = chebyshev_basis_matrix(N_per, tau_loc)
T_basis_loc_ca = ca.DM(T_basis_loc)

# --- Differentiation Matrix ---
def chebyshev_diff_matrix(N, tau):
    D = np.zeros((N + 1, N + 1))
    
    # Weights for the differentiation matrix
    w = np.ones(N + 1)
    w[0] = w[N] = 0.5
    
    # Differentiation matrix formula for CGL nodes
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = w[j] / w[i] * (-1)**(i + j) / (tau[i] - tau[j])
            elif i == j and i != 0 and i != N:
                D[i, j] = -tau[i] / (2 * (1 - tau[i]**2))
            elif i == 0:
                D[0, 0] = (2*N**2 + 1)/6
            elif i == N:
                D[N, N] = -(2*N**2 + 1)/6
    
    return D

D_loc = chebyshev_diff_matrix(N_per, tau_loc)
D_loc_ca = ca.DM(D_loc)

# --- Planar VTOL Vehicle Dynamics ---
def f_dynamics(x, u):
    """Planar VTOL vehicle dynamics
    State vector: x = [px, pz, phi, vx, vz, phi_dot]
    Control vector: u = [u1, u2] (thrust forces from two propellers)
    """
    phi, vx, vz, phi_dot = x[2], x[3], x[4], x[5]
    u1, u2 = u[0], u[1]  # Thrust forces from propellers
    
    # Trigonometric functions
    cos_phi = ca.cos(phi)
    sin_phi = ca.sin(phi)
    
    # Position derivatives (velocities)
    px_dot = vx
    pz_dot = vz
    
    # Orientation derivative
    phi_dot_out = phi_dot
    
    # Translational accelerations
    vx_dot = -(u1 + u2) * sin_phi / m
    vz_dot = (u1 + u2) * cos_phi / m - g
    
    # Rotational acceleration
    phi_ddot = l * (u1 - u2) / J
    
    return ca.vertcat(px_dot, pz_dot, phi_dot_out, vx_dot, vz_dot, phi_ddot)

print("=== MULTI-INTERVAL CHEBYSHEV PSEUDOSPECTRAL OPTIMIZATION - VTOL ===")
print(f"Using S={S_intervals} intervals, {N_per+1} CGL nodes per interval (total nodes {S_intervals*N_per+1})")
print(f"Local basis condition number: {np.linalg.cond(T_basis_loc):.2e}")

# --- NLP Setup with Modal Basis Constraints ---
opti = ca.Opti()

# --- Multi-interval decision variables ---
X_blocks = []  # list of (n_x, N_per+1)
U_blocks = []  # list of (n_u, N_per+1)
for _ in range(S_intervals):
    X_blocks.append(opti.variable(n_x, N_per + 1))
    U_blocks.append(opti.variable(n_u, N_per + 1))

T_final = opti.variable()  # Final time

# (Objective moved to multi-interval quadrature section below)

# --- Boundary Conditions ---
# Initial: hover at origin with 60° attitude
x_start = ca.DM([-0.5, 0.5, -np.pi/3, 0.0, 0.0, 0.0])
# Final: hover at (1.75,1.75) with 0° attitude
x_final = ca.DM([1.75, 1.75, 0.0, 0.0, 0.0, 0.0])

# Initial and final state constraints across blocks
opti.subject_to(X_blocks[0][:, 0] == x_start)
opti.subject_to(X_blocks[-1][:, -1] == x_final)
opti.subject_to(T_final >= 8.0)  # Minimum time constraint
opti.subject_to(T_final <= 10.0)

# --- Continuity constraints between intervals ---
for s in range(S_intervals - 1):
    opti.subject_to(X_blocks[s][:, -1] == X_blocks[s + 1][:, 0])
    # Enforce control continuity as well to keep u consistent between intervals
    opti.subject_to(U_blocks[s][:, -1] == U_blocks[s + 1][:, 0])

# --- Pseudospectral Dynamic Constraints (per interval) ---
scale = T_final / (2 * S_intervals)
for s in range(S_intervals):
    Xs = X_blocks[s]
    Us = U_blocks[s]
    for k in range(N_per + 1):
        # Drop last-row collocation at internal boundaries
        if (k == N_per) and (s < S_intervals - 1):
            continue
        # Left side: dX/dtau approximated by D_loc @ X
        X_dot_tau = ca.mtimes(D_loc_ca[k, :], Xs.T)
        X_dot_tau = X_dot_tau.T  # make it (n_x,)
        # Right side: scaled dynamics
        f_val = f_dynamics(Xs[:, k], Us[:, k])
        opti.subject_to(X_dot_tau == scale * f_val)

# --- Path Constraints ---
u_max = 4.0
pos_max = 2.0
vel_max = 5.0
angle_max = np.pi/2
theta_dot_max = 10.0

for s in range(S_intervals):
    Xs = X_blocks[s]
    Us = U_blocks[s]
    # Control bounds at each node (element-wise)
    opti.subject_to(opti.bounded(0, Us, u_max))
    # State bounds (element-wise)
    opti.subject_to(opti.bounded(-pos_max, Xs[0, :], pos_max))
    opti.subject_to(opti.bounded(0, Xs[1, :], pos_max))
    opti.subject_to(opti.bounded(-angle_max, Xs[2, :], angle_max))
    opti.subject_to(opti.bounded(-vel_max, Xs[3, :], vel_max))
    opti.subject_to(opti.bounded(-vel_max, Xs[4, :], vel_max))
    opti.subject_to(opti.bounded(-theta_dot_max, Xs[5, :], theta_dot_max))

# --- Initial Guess ---
T_init = 5.0
opti.set_initial(T_final, T_init)

for s in range(S_intervals):
    X_guess = np.zeros((n_x, N_per + 1))
    # Linear interpolation per block
    alpha0 = s / S_intervals
    alpha1 = (s + 1) / S_intervals
    x0 = (1 - alpha0) * x_start + alpha0 * x_final
    x1 = (1 - alpha1) * x_start + alpha1 * x_final
    for i in range(n_x):
        X_guess[i, :] = np.linspace(float(x0[i]), float(x1[i]), N_per + 1)
    U_guess = np.ones((n_u, N_per + 1)) * m * g / 2
    opti.set_initial(X_blocks[s], X_guess)
    opti.set_initial(U_blocks[s], U_guess)

# --- Objective Function (multi-interval quadrature) ---
# J = T_final + (T_final/(2S)) * sum_{s=0..S-1} sum_{j=1..N} c_j * ||U^s_j||^2
def L(u):
    return ca.mtimes(u.T, u)  # control effort

cost = T_final
for s in range(S_intervals):
    Us = U_blocks[s]
    # Determine node range: exclude last seam node for internal intervals to avoid bias
    j_end_effort = N_per if s <= S_intervals - 1 else N_per - 1
    # Control effort term (uniform weights)
    for j in range(1, j_end_effort + 1):  # j=1..j_end_effort
        cost += (T_final / (2 * S_intervals)) * 10 * L(Us[:, j])
    # Control rate smoothing: approximate du/dt via local Chebyshev differentiation
    # du/dtau ≈ D_loc * U, so du/dt = (2S/T) * du/dtau
    # dU_dt_nodes shape: (N_per+1, n_u). Each row is the rate of change for all control inputs at a node.
    # The smoothing cost sums the squared rates for all control inputs at each node.
    dU_dt_nodes = (2 * S_intervals) / T_final * ca.mtimes(D_loc_ca, Us.T)  # (N_per+1, n_u)
    j_end_rate = N_per if s <= S_intervals - 1 else N_per - 1
    for j in range(1, j_end_rate + 1):  # mirror interior-node selection
        # ca.sumsqr(dU_dt_nodes[j, :]) sums over all control inputs at node j
        cost += LAMBDA_U_RATE * (T_final / (2 * S_intervals)) * ca.sumsqr(dU_dt_nodes[j, :])


opti.minimize(cost)

# --- Solver Configuration ---
opti.solver('ipopt', {'print_time': False}, {
    'tol': 1e-6,
    'max_iter': 2000,
    'linear_solver': 'mumps'
})

# --- Solve Optimization ---
try:
    print(f"\n🚀 Starting optimization with S={S_intervals}, N={N_per} (total nodes {S_intervals*N_per+1})...")
    sol = opti.solve()
    print("\n✅ Optimization successful!")
    
    # Extract results
    T_opt = sol.value(T_final)
    # Assemble global trajectories by stitching blocks (skip shared starts)
    total_nodes = S_intervals * N_per + 1
    X_opt = np.zeros((n_x, total_nodes))
    U_opt = np.zeros((n_u, total_nodes))
    idx = 0
    for s in range(S_intervals):
        Xs = sol.value(X_blocks[s])
        Us = sol.value(U_blocks[s])
        if s == 0:
            X_opt[:, idx:idx + N_per + 1] = Xs
            U_opt[:, idx:idx + N_per + 1] = Us
            idx += N_per + 1
        else:
            X_opt[:, idx:idx + N_per] = Xs[:, 1:]
            U_opt[:, idx:idx + N_per] = Us[:, 1:]
            idx += N_per
    
    # (Coefficient analysis omitted in multi-interval nodal formulation)
    
    # Map to real time per interval
    t_opt = np.zeros(total_nodes)
    idx = 0
    for s in range(S_intervals):
        t0s = (T_opt / S_intervals) * s
        # Map CGL nodes to increasing time within the interval: t = t0 + (T/S)*(1 - tau)/2
        local_times = t0s + 0.5 * (1.0 - tau_loc) * (T_opt / S_intervals)
        if s == 0:
            t_opt[idx:idx + N_per + 1] = local_times
            idx += N_per + 1
        else:
            t_opt[idx:idx + N_per] = local_times[1:]
            idx += N_per
    print(f"\n🎯 Optimization Results:")
    print(f"   Optimal time: {T_opt:.4f} s")
    print(f"   Final position: ({X_opt[0, -1]:.4f}, {X_opt[1, -1]:.4f}) m")
    print(f"   Final attitude: {X_opt[2, -1]*180/np.pi:.3f}°")

    # Diagnostics: continuity across interval seams
    # seam_max_err = 0.0
    # for s in range(1, S_intervals):
    #     seam_idx = s * N_per
    #     err = np.linalg.norm(X_opt[:, seam_idx] - X_opt[:, seam_idx], ord=np.inf)
    #     seam_max_err = max(seam_max_err, err)
    # print(f"   Continuity check at seams (states): max ||jump||_inf = {seam_max_err:.2e}")
    
except RuntimeError as e:
    print(f"\n❌ Optimization failed: {e}")
    print("   Try adjusting solver settings or reducing N")

#%%
# --- Visualization ---
import matplotlib.pyplot as plt

if 'sol' in locals():
    plt.figure(figsize=(15, 12))
    
    # State trajectories
    state_names = ['Position X [m]', 'Position Z [m]', 'Attitude [deg]', 
                   'Velocity X [m/s]', 'Velocity Z [m/s]', 'Angular Velocity [deg/s]']
    
    for i in range(6):
        plt.subplot(3, 3, i+1)
        if i == 2 or i == 5:  # Convert angles to degrees
            plt.plot(t_opt, X_opt[i, :] * 180/np.pi, linewidth=2)
        else:
            plt.plot(t_opt, X_opt[i, :], linewidth=2)
        plt.title(state_names[i])
        plt.xlabel('Time [s]')
        plt.ylabel(state_names[i])
        plt.grid(True)
        # Mark interval boundaries
        for s in range(1, S_intervals):
            t_seam = s * T_opt / S_intervals
            plt.axvline(t_seam, color='k', alpha=0.2, linestyle='--')
    
    # Control inputs
    plt.subplot(3, 3, 7)
    plt.plot(t_opt, U_opt[0, :], 'b-', linewidth=2, label='u1 (Prop 1)')
    plt.plot(t_opt, U_opt[1, :], 'r-', linewidth=2, label='u2 (Prop 2)')
    plt.title('Thrust Forces')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(True)
    plt.legend()
    for s in range(1, S_intervals):
        t_seam = s * T_opt / S_intervals
        plt.axvline(t_seam, color='k', alpha=0.2, linestyle='--')
    
    # Trajectory in x-z plane with VTOL snapshots
    plt.subplot(3, 3, 8)
    plt.plot(X_opt[0, :], X_opt[1, :], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
    # Visualize interval boundaries on the path by marking seam nodes
    seam_x = []
    seam_z = []
    for s in range(1, S_intervals):
        seam_idx = s * N_per
        seam_x.append(X_opt[0, seam_idx])
        seam_z.append(X_opt[1, seam_idx])
    if len(seam_x) > 0:
        plt.scatter(seam_x, seam_z, c='k', s=35, marker='x', label='Seams')
    
    # Draw VTOL at 10 snapshots
    n_snapshots = 10
    snapshot_indices = np.linspace(0, S_intervals * N_per, n_snapshots, dtype=int)
    
    for i, idx in enumerate(snapshot_indices):
        x_pos = X_opt[0, idx]
        z_pos = X_opt[1, idx]
        phi = X_opt[2, idx]  # attitude angle
        
        # VTOL body parameters for visualization
        body_length = 0.15
        prop_offset = 0.12
        
        # Body endpoints
        body_x = [x_pos - body_length/2 * np.cos(phi), x_pos + body_length/2 * np.cos(phi)]
        body_z = [z_pos - body_length/2 * np.sin(phi), z_pos + body_length/2 * np.sin(phi)]
        
        # Propeller positions
        prop1_x = x_pos - prop_offset * np.cos(phi)
        prop1_z = z_pos - prop_offset * np.sin(phi)
        prop2_x = x_pos + prop_offset * np.cos(phi)
        prop2_z = z_pos + prop_offset * np.sin(phi)
        
        # Color based on time (blue to red)
        color = plt.cm.coolwarm(i / (n_snapshots - 1))
        
        # Draw VTOL body
        plt.plot(body_x, body_z, color=color, linewidth=3, alpha=0.8)
        
        # Draw propellers as small circles
        plt.scatter([prop1_x, prop2_x], [prop1_z, prop2_z], 
                   c=[color], s=30, marker='o', alpha=0.8)
        
        # Add thrust vectors (scaled by thrust magnitude)
        thrust_scale = 0.05
        thrust1_mag = U_opt[0, idx] * thrust_scale
        thrust2_mag = U_opt[1, idx] * thrust_scale
        
        # Thrust direction is perpendicular to body (upward relative to VTOL)
        thrust_x_dir = -np.sin(phi)
        thrust_z_dir = np.cos(phi)
        
        # Draw thrust vectors
        plt.arrow(prop1_x, prop1_z, thrust_x_dir * thrust1_mag, thrust_z_dir * thrust1_mag,
                 head_width=0.02, head_length=0.02, fc=color, ec=color, alpha=0.6)
        plt.arrow(prop2_x, prop2_z, thrust_x_dir * thrust2_mag, thrust_z_dir * thrust2_mag,
                 head_width=0.02, head_length=0.02, fc=color, ec=color, alpha=0.6)
    
    # Mark start and end points
    plt.scatter(X_opt[0, 0], X_opt[1, 0], c='green', s=150, label='Start', marker='o', zorder=10)
    plt.scatter(X_opt[0, -1], X_opt[1, -1], c='red', s=150, label='End', marker='s', zorder=10)
    
    plt.title('VTOL Trajectory with Vehicle Snapshots')
    plt.xlabel('X Position [m]')
    plt.ylabel('Z Position [m]')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    
    # (Coefficient magnitude plot omitted for multi-interval nodal formulation)
    
    plt.tight_layout()
    plt.show()
    
    
#%%
# Animation of VTOL trajectory
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_title('VTOL Trajectory Animation')
ax1.set_xlabel('X Position [m]')
ax1.set_ylabel('Z Position [m]')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# Set plot limits with some margin
x_margin = 2.5
z_margin = 2.5
ax1.set_xlim(np.min(X_opt[0, :]) - x_margin, np.max(X_opt[0, :]) + x_margin)
ax1.set_ylim(np.min(X_opt[1, :]) - z_margin, np.max(X_opt[1, :]) + z_margin)

# Initialize animation elements
line_traj, = ax1.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
body_line, = ax1.plot([], [], 'k-', linewidth=4)
prop1_circle = plt.Circle((0, 0), 0.03, color='red', fill=True)
prop2_circle = plt.Circle((0, 0), 0.03, color='red', fill=True)
ax1.add_patch(prop1_circle)
ax1.add_patch(prop2_circle)

# Thrust arrows
thrust1_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2))
thrust2_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='orange', lw=2))

# Text displays
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Mark start and end points
ax1.scatter(X_opt[0, 0], X_opt[1, 0], c='green', s=100, label='Start', marker='o', zorder=10)
ax1.scatter(X_opt[0, -1], X_opt[1, -1], c='red', s=100, label='End', marker='s', zorder=10)
ax1.legend()

def animate(frame, X_opt, U_opt, t_opt,
            line_traj, body_line, prop1_circle, prop2_circle,
            thrust1_arrow, thrust2_arrow, time_text):
    # Update trajectory line (show path up to current frame)
    line_traj.set_data(X_opt[0, :frame+1], X_opt[1, :frame+1])
    
    # Current VTOL position and attitude
    x_pos = X_opt[0, frame]
    z_pos = X_opt[1, frame]
    phi = X_opt[2, frame]
    
    # VTOL body parameters
    body_length = 0.12
    prop_offset = 0.08
    
    # Body endpoints
    body_x = [x_pos - body_length/2 * np.cos(phi), x_pos + body_length/2 * np.cos(phi)]
    body_z = [z_pos - body_length/2 * np.sin(phi), z_pos + body_length/2 * np.sin(phi)]
    body_line.set_data(body_x, body_z)
    
    # Propeller positions
    prop1_x = x_pos - prop_offset * np.cos(phi)
    prop1_z = z_pos - prop_offset * np.sin(phi)
    prop2_x = x_pos + prop_offset * np.cos(phi)
    prop2_z = z_pos + prop_offset * np.sin(phi)
    
    # Update propeller circles
    prop1_circle.center = (prop1_x, prop1_z)
    prop2_circle.center = (prop2_x, prop2_z)
    
    # Thrust vectors (scaled by thrust magnitude)
    thrust_scale = 0.1
    thrust1_mag = U_opt[0, frame] * thrust_scale
    thrust2_mag = U_opt[1, frame] * thrust_scale
    
    # Thrust direction (perpendicular to body, upward relative to VTOL)
    thrust_x_dir = -np.sin(phi)
    thrust_z_dir = np.cos(phi)
    
    # Update thrust arrows
    thrust1_arrow.set_position((prop1_x, prop1_z))
    thrust1_arrow.xy = (prop1_x + thrust_x_dir * thrust1_mag, 
                prop1_z + thrust_z_dir * thrust1_mag)
    thrust1_arrow.xytext = (prop1_x, prop1_z)
    
    thrust2_arrow.set_position((prop2_x, prop2_z))
    thrust2_arrow.xy = (prop2_x + thrust_x_dir * thrust2_mag, 
                prop2_z + thrust_z_dir * thrust2_mag)
    thrust2_arrow.xytext = (prop2_x, prop2_z)
    
    # Update time and status text
    current_time = t_opt[frame]
    time_text.set_text(f'Time: {current_time:.2f}s\n'
                        f'Thrust 1: {U_opt[0, frame]:.2f}N\n'
                        f'Thrust 2: {U_opt[1, frame]:.2f}N\n'
                        f'Attitude: {phi*180/np.pi:.1f}°')
    
    return line_traj, body_line, prop1_circle, prop2_circle, thrust1_arrow, thrust2_arrow, time_text

# Create animation
# Subsample frames for faster rendering while keeping smooth playback
total_nodes = S_intervals * N_per + 1
target_fps = 30  # Realistic frame rate
total_frames = int(T_opt * target_fps)  # e.g., 4s * 30fps = 120 frames
frame_indices = np.linspace(0, total_nodes - 1, total_frames, dtype=int)

# Milliseconds per frame for real-time playback
ms_per_frame = 1000.0 / target_fps

anim = FuncAnimation(
    fig,
    animate,
    frames=frame_indices,
    interval=ms_per_frame,
    blit=False,
    repeat=True,
    fargs=(X_opt, U_opt, t_opt,
           line_traj, body_line, prop1_circle, prop2_circle,
           thrust1_arrow, thrust2_arrow, time_text)
)

plt.tight_layout()
plt.show()

# Optional: Save animation as GIF
# anim.save('vtol_trajectory.gif', writer='pillow', fps=10)

#%%save x_opt, u_opt, t_opt to npz file
# Save optimization results
if 'sol' in locals():
    np.savez('vtol_optimization_results.npz', X_opt=X_opt, U_opt=U_opt, t_opt=t_opt)
    print("\n💾 Optimization results saved to 'vtol_optimization_results.npz'")