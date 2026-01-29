import casadi as ca
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpc_utils import cgl_nodes, create_mpc_controller, generate_reference_path, rk4_step
import time

# load optimization results
data = np.load('vtol_optimization_results.npz')
X_opt = data['X_opt']
U_opt = data['U_opt']
t_opt = data['t_opt']
T_opt = t_opt[-1]


# --- User-Defined Parameters ---
n_x = 6 # Number of states (px, pz, phi, vx, vz, phi_dot)
n_u = 2 # Number of controls (u1, u2 - thrust forces from two propellers)

# VTOL physical parameters (from the document)
m = 0.486  # Mass [kg]
J = 0.00383  # Moment of inertia [kg⋅m²]
l = 0.25   # Distance between propellers and center of mass [m]
g = 9.81   # Gravity [m/s²]

T_horizon = 1.0  # MPC prediction horizon [s] (longer to see further ahead)
N_intervals = 5  # Number of Chebyshev intervals in the horizon (more intervals = better approximation)
N_per = 5        # Chebyshev nodes per interval (CGL, N_per+1 nodes)
N_mpc = N_intervals * N_per  # Total number of nodes in MPC horizon
dt_mpc = T_horizon / N_intervals  # Time step per interval

# Scaling flag: True = use normalized variables [0,1], False = use physical variables
USE_SCALING = False

# Create MPC controller
mpc_controller, x_current, x_ref_traj, u_ref_traj, X_mpc, U_mpc = create_mpc_controller(T_horizon, N_intervals, N_per, USE_SCALING)

# Simulation parameters
t_sim_end = T_opt
dt_sim = 0.005  # Simulation time step
t_sim = np.arange(0, t_sim_end, dt_sim)
N_sim = len(t_sim)

# Generate reference trajectory
x_ref_full, u_ref_full = generate_reference_path(t_sim)

# Initialize simulation
x_sim = np.zeros((n_x, N_sim))
u_sim = np.zeros((n_u, N_sim))
# Start exactly at the reference initial state to ensure consistent tracking
# (was [0, 0, 0, 0, 0, 0], which mismatched the optimized trajectory start)
x_sim[:, 0] = [-0.25, 1, 0, 0, 0, 0] # 1.1 * X_opt[:, 0]

# Print reference trajectory info
print(f"📍 Reference trajectory info:")
print(f"   Start position: [{X_opt[0, 0]:.3f}, {X_opt[1, 0]:.3f}] m")
print(f"   End position: [{X_opt[0, -1]:.3f}, {X_opt[1, -1]:.3f}] m")
print(f"   Total distance: {np.linalg.norm(X_opt[:2, -1] - X_opt[:2, 0]):.3f} m")
print(f"   Start velocity: [{X_opt[3, 0]:.3f}, {X_opt[4, 0]:.3f}] m/s")
print(f"   Start angle: {X_opt[2, 0]:.3f} rad = {np.rad2deg(X_opt[2, 0]):.1f} deg")
print(f"   Start control: [{U_opt[0, 0]:.3f}, {U_opt[1, 0]:.3f}] N")

print(f"   Simulation time: {t_sim_end}s with {N_sim} steps")
print(f"   MPC horizon: {N_mpc} steps, dt_mpc = {dt_mpc}s")
print(f"   Variable scaling: {'ENABLED (normalized [0,1])' if USE_SCALING else 'DISABLED (physical units)'}")

# --- Simulation Loop ---
# Control update rate: 50 Hz (0.02s period)
control_dt = 0.1  # seconds
control_steps = int(control_dt / dt_sim)  # Number of sim steps per control update

print(f"   Control frequency: {1/control_dt:.0f} Hz (updates every {control_dt}s = {control_steps} sim steps)")

mpc_solve_times = []
mpc_failure_count = 0  # Track consecutive solver failures
max_failures = 10      # Break simulation if failures exceed this
# Initialize previous solution for warm-starting
X_opt_mpc = None
U_opt_mpc = None
u_current = u_ref_full[:, 0]  # Hold control between MPC solves

# Store MPC predictions for animation
mpc_predictions = []  # List of (time, X_predicted) tuples

for k in range(N_sim - 1):
    # Only solve MPC every control_steps
    if k % control_steps == 0:
        print(f"   Step {k}/{N_sim-1} (t = {t_sim[k]:.2f}s) - Solving MPC...")
        
        # Set current state
        mpc_controller.set_value(x_current, x_sim[:, k])

        # Create reference trajectory over MPC horizon with proper time sampling
        # MPC nodes are at CGL-based times within each interval
        t_current = t_sim[k]
        x_ref_horizon = np.zeros((n_x, N_mpc + 1))
        u_ref_horizon = np.zeros((n_u, N_mpc + 1))
        
        # Sample reference at each MPC node's actual time
        # Build global time array for all MPC nodes
        node_counter = 0
        for i_interval in range(N_intervals):
            # Local CGL nodes in [-1, 1] (ascending order)
            tau_loc = cgl_nodes(N_per)  # [-1, ..., +1]
            
            # Map to physical time within this interval
            t_interval_start = t_current + i_interval * dt_mpc
            
            # Determine how many nodes to process
            # First interval: all N_per+1 nodes
            # Subsequent intervals: skip first node (shared with previous interval), process N_per nodes
            j_start = 1 if i_interval > 0 else 0
            
            for j_loc in range(j_start, N_per + 1):
                # CGL tau goes -1 to +1, map to time going forward
                # tau = -1 -> t_start, tau = +1 -> t_end
                t_node = t_interval_start + (1 + tau_loc[j_loc]) / 2 * dt_mpc
                t_node = np.clip(t_node, 0, T_opt)  # Clamp to reference range
                
                # Interpolate reference at this time
                for state_idx in range(n_x):
                    x_ref_horizon[state_idx, node_counter] = np.interp(t_node, t_opt, X_opt[state_idx, :])
                for control_idx in range(n_u):
                    u_ref_horizon[control_idx, node_counter] = np.interp(t_node, t_opt, U_opt[control_idx, :])
                
                node_counter += 1
        
        mpc_controller.set_value(x_ref_traj, x_ref_horizon)
        mpc_controller.set_value(u_ref_traj, u_ref_horizon)
        
        # Initial guess (warm start) - use reference trajectory!
        if k == 0 or X_opt_mpc is None:
            # Use reference trajectory as initial guess
            X_init_mpc = x_ref_horizon
            U_init_mpc = u_ref_horizon
        else:
            # Shift previous solution, but blend with reference
            X_shifted = np.column_stack([X_opt_mpc[:, 1:], X_opt_mpc[:, -1]])
            U_shifted = np.column_stack([U_opt_mpc[:, 1:], U_opt_mpc[:, -1]])
            # Use 50% reference, 50% shifted previous solution for better tracking
            X_init_mpc = 1 * x_ref_horizon + 0.0 * X_shifted
            U_init_mpc = 1 * u_ref_horizon + 0.0 * U_shifted

        mpc_controller.set_initial(X_mpc, X_init_mpc)
        mpc_controller.set_initial(U_mpc, U_init_mpc)
        
        # Solve MPC
        try:
            solve_start = time.time()
            sol_mpc = mpc_controller.solve()
            solve_time = time.time() - solve_start
            mpc_solve_times.append(solve_time)
            
            # Extract solution
            X_opt_mpc = sol_mpc.value(X_mpc)
            U_opt_mpc = sol_mpc.value(U_mpc)
            
            # Store MPC prediction for animation
            mpc_predictions.append((t_sim[k], X_opt_mpc.copy(), x_ref_horizon.copy()))
            
            # Update control (will be held for next control_steps)
            u_current = U_opt_mpc[:, 0]
            
            # Reset failure counter on successful solve
            mpc_failure_count = 0
        
        except Exception as e:
            print(f"   ❌ MPC solve failed at step {k}: {e}")
            mpc_failure_count += 1
            
            # Check if too many failures
            if mpc_failure_count >= max_failures:
                print(f"\n🛑 STOPPING: MPC solver failed {mpc_failure_count} times (threshold: {max_failures})")
                print(f"   Last successful solve at t = {t_sim[k]:.2f}s")
                # Trim arrays to actual simulation length
                N_sim = k + 1
                x_sim = x_sim[:, :N_sim]
                u_sim = u_sim[:, :N_sim]
                t_sim = t_sim[:N_sim]
                x_ref_full = x_ref_full[:, :N_sim]
                u_ref_full = u_ref_full[:, :N_sim]
                break
            
            # Fallback to reference control if solver fails
            if k == 0:
                # Use reference control at current time
                u_current = u_ref_horizon[:, 0]
            # else: keep previous u_current (already set from last successful solve)
    
    # Apply control (ZOH - held from last MPC solve)
    u_sim[:, k] = u_current
    
    # Simulate system forward (using RK4 integration from mpc_utils)
    x_sim[:, k+1] = rk4_step(x_sim[:, k], u_sim[:, k], dt_sim)

print(f"\n✅ MPC Simulation completed!")
if len(mpc_solve_times) > 0:
    print(f"   Average solve time: {np.mean(mpc_solve_times):.4f}s")
    print(f"   Max solve time: {np.max(mpc_solve_times):.4f}s")
    print(f"   Number of successful solves: {len(mpc_solve_times)}")
else:
    print(f"   ⚠️ WARNING: MPC solver never succeeded!")
    print(f"   All {mpc_failure_count} MPC attempts failed")
final_pos_error = np.linalg.norm(x_sim[:2, -1] - x_ref_full[:2, -1])
print(f"   Final position error: {final_pos_error:.4f} m")

# Visualization of MPC results
plt.figure(figsize=(15, 10))

# Path tracking in x-z plane
plt.subplot(2, 3, 1)
plt.plot(x_ref_full[0, :], x_ref_full[1, :], 'r--', linewidth=2, label='Reference')
plt.plot(x_sim[0, :], x_sim[1, :], 'b-', linewidth=2, label='MPC Tracking')
plt.scatter(x_sim[0, 0], x_sim[1, 0], c='green', s=100, label='Start', marker='o')
plt.xlabel('X Position [m]')
plt.ylabel('Z Position [m]')
plt.title('Path Tracking Performance')
plt.legend()
plt.grid(True)
plt.axis('equal')

# State tracking errors
state_names = ['X pos', 'Z pos', 'Attitude', 'X vel', 'Z vel', 'Ang vel']
for i in range(5):  # Only plot first 5 states to fit in 2x3 layout
    plt.subplot(2, 3, i+2)
    error = x_sim[i, :] - x_ref_full[i, :]
    if i == 2:  # Convert angles to degrees
        error = error * 180/np.pi
        plt.ylabel('Error [deg]')
    else:
        plt.ylabel('Error [m or m/s]')
    
    plt.plot(t_sim, error, linewidth=2)
    plt.title(f'{state_names[i]} Tracking Error')
    plt.xlabel('Time [s]')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Control inputs comparison
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t_sim[:-1], u_sim[0, :-1], 'b-', linewidth=2, label='MPC u1')
plt.plot(t_sim, u_ref_full[0, :], 'r--', linewidth=2, label='Reference u1')
plt.ylabel('Thrust 1 [N]')
plt.title('Control Input 1')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t_sim[:-1], u_sim[1, :-1], 'b-', linewidth=2, label='MPC u2')
plt.plot(t_sim, u_ref_full[1, :], 'r--', linewidth=2, label='Reference u2')
plt.ylabel('Thrust 2 [N]')
plt.title('Control Input 2')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
# Create time array for MPC updates (every control_dt seconds)
t_mpc_updates = np.arange(0, len(mpc_solve_times)) * control_dt
plt.plot(t_mpc_updates, np.array(mpc_solve_times), 'g-', linewidth=2, marker='o')
plt.ylabel('Solve Time [s]')
plt.xlabel('Time [s]')
plt.title('MPC Computation Time')
plt.grid(True)

plt.subplot(2, 2, 4)
# Tracking performance metrics
tracking_error = np.linalg.norm(x_sim[:2, :] - x_ref_full[:2, :], axis=0)
plt.plot(t_sim, tracking_error, 'k-', linewidth=2)
plt.ylabel('Position Error [m]')
plt.xlabel('Time [s]')
plt.title('Position Tracking Error')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\n📊 MPC Performance Metrics:")
print(f"   Mean position error: {np.mean(tracking_error):.4f} m")
print(f"   Max position error: {np.max(tracking_error):.4f} m")
print(f"   RMS position error: {np.sqrt(np.mean(tracking_error**2)):.4f} m")
print(f"   Real-time factor: {dt_sim/np.mean(mpc_solve_times):.1f}x")

# --- Animation with MPC Horizon Visualization ---
print("\n🎬 Creating animation...")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot full reference trajectory
ax.plot(x_ref_full[0, :], x_ref_full[1, :], 'r--', linewidth=1.5, alpha=0.5, label='Reference Path')
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Z Position [m]')
ax.set_title('VTOL MPC Tracking with Horizon Visualization')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend(loc='upper right')

# Initialize plot elements
vtol_body, = ax.plot([], [], 'b-', linewidth=3, label='VTOL Body')
vtol_props, = ax.plot([], [], 'ko', markersize=8, label='Propellers')
trajectory, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Actual Trajectory')
mpc_horizon, = ax.plot([], [], 'g--', linewidth=2, alpha=0.8, label='MPC Horizon')
mpc_ref_horizon, = ax.plot([], [], 'm:', linewidth=2, alpha=0.6, label='Reference Horizon')
time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def draw_vtol(x_pos, z_pos, phi):
    """Draw VTOL vehicle as a line with propellers at ends"""
    # Body endpoints in local frame
    x_local = np.array([-l/2, l/2])
    z_local = np.array([0, 0])
    
    # Rotate and translate to world frame
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_world = x_pos + cos_phi * x_local - sin_phi * z_local
    z_world = z_pos + sin_phi * x_local + cos_phi * z_local
    
    return x_world, z_world

def init():
    """Initialize animation"""
    vtol_body.set_data([], [])
    vtol_props.set_data([], [])
    trajectory.set_data([], [])
    mpc_horizon.set_data([], [])
    mpc_ref_horizon.set_data([], [])
    time_text.set_text('')
    return vtol_body, vtol_props, trajectory, mpc_horizon, mpc_ref_horizon, time_text

def animate(frame):
    """Animation update function"""
    # Frame corresponds to simulation step
    # Downsample for animation speed (show every 10th frame)
    k = frame * 10
    if k >= len(t_sim):
        k = len(t_sim) - 1
    
    # Draw VTOL at current position
    x_pos, z_pos, phi = x_sim[0, k], x_sim[1, k], x_sim[2, k]
    x_body, z_body = draw_vtol(x_pos, z_pos, phi)
    vtol_body.set_data(x_body, z_body)
    vtol_props.set_data(x_body, z_body)
    
    # Draw trajectory up to current time
    trajectory.set_data(x_sim[0, :k+1], x_sim[1, :k+1])
    
    # Find and draw MPC horizon for this time
    # Find closest MPC prediction time
    current_time = t_sim[k]
    mpc_idx = None
    for i, (t_mpc, _, _) in enumerate(mpc_predictions):
        if abs(t_mpc - current_time) < control_dt / 2:
            mpc_idx = i
            break
    
    if mpc_idx is not None:
        _, X_pred, X_ref_pred = mpc_predictions[mpc_idx]
        mpc_horizon.set_data(X_pred[0, :], X_pred[1, :])
        mpc_ref_horizon.set_data(X_ref_pred[0, :], X_ref_pred[1, :])
    else:
        mpc_horizon.set_data([], [])
        mpc_ref_horizon.set_data([], [])
    
    time_text.set_text(f'Time: {current_time:.2f}s / {t_sim_end:.2f}s')
    
    return vtol_body, vtol_props, trajectory, mpc_horizon, mpc_ref_horizon, time_text

# Create animation
n_frames = len(t_sim) // 10  # Downsample for reasonable animation length
anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                     interval=50, blit=True, repeat=True)

plt.legend(loc='upper right')
plt.show()

print("✅ Animation complete!")

# %%





