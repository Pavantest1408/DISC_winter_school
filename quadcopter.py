import numpy as np
import scipy.linalg as la
import cvxpy as cp

# SYSTEM DEFINITION 
class PlanarQuadrotor:
    def __init__(self):
        self.m = 0.486  # kg
        self.J = 0.00383 # kg m^2
        self.l = 0.25   # m
        self.g = 9.81
        
        # Control Matrix B (Constant)
        self.B = np.array([
            [0, 0], [0, 0], [0, 0], [0, 0],
            [1/self.m, 1/self.m],
            [self.l/self.J, -self.l/self.J]
        ])

    def f(self, x):
        #Nominal Drift Dynamics
        px, pz, phi, vx, vz, phi_dot = x
        return np.array([
            vx*np.cos(phi) - vz*np.sin(phi),
            vx*np.sin(phi) + vz*np.cos(phi),
            phi_dot,
            vz*phi_dot - self.g*np.sin(phi),
            -vx*phi_dot - self.g*np.cos(phi),
            0
        ])

    def get_calibration_data(self, x, u, t):
    
        
        # Nominal
        dx_nom = self.f(x) + self.B @ u
        
        # Reality (Add random wind noise)
        wind_force = 0.5 * np.sin(2*t) + np.random.normal(0, 0.2)
        d = np.zeros(6)
        d[3] = wind_force / self.m
        d[4] = wind_force / self.m
        
        return dx_nom + d

    def get_linearized_dynamics(self):
        #Jacobian A 
        A = np.zeros((6, 6))
        A[0, 3] = 1; A[1, 4] = 1; A[2, 5] = 1
        A[3, 2] = -self.g
        return A

# CALCULATE WIND BOUND (d_bar)
def calculate_wind_bound(sys):    
    # Collect 1000 data points
    N = 1000
    scores = []
    x = np.zeros(6)
    u = np.array([sys.m * sys.g / 2, sys.m * sys.g / 2]) # hovering 
    
    for i in range(N):
        t = i * 0.01
        dx_true = sys.get_calibration_data(x, u, t)
        dx_nom = sys.f(x) + sys.B @ u
        
        # Score = Difference between Reality and Model
        score = np.linalg.norm(dx_true - dx_nom)
        scores.append(score)
        x = x + dx_true * 0.01

    # Sort and Pick the 95th Percentile
    # Reference: Slide 23 & 34 of Part 3
    delta = 0.05
    k = int(np.ceil((1 - delta) * (N + 1)))
    sorted_scores = np.sort(scores)
    d_bar = sorted_scores[k-1]
    
    print(f" Processed {N} samples.")
    print(f" Calculated 95% Confidence Bound (d_bar): {d_bar:.4f}")
    return d_bar

# CALCULATE CONTRACTION METRIC (M)
def calculate_metric(sys, alpha=1.0):
    print("\n Solving Convex Optimization for Metric M")
    
    # Variables
    W = cp.Variable((6, 6), symmetric=True)
    A = sys.get_linearized_dynamics()
    rho = 1.0
    
    # Constraints 
    constraints = [W >> 0.1 * np.eye(6)] # Positive Definite
    
    # A*W + W*A^T - 2*rho*B*B^T <= -2*alpha*W
    BBT = sys.B @ sys.B.T
    LMI = A @ W + W @ A.T - 2 * rho * BBT + 2 * alpha * W
    constraints.append(LMI << 0)
    
    # Solve
    prob = cp.Problem(cp.Minimize(0), constraints) # here cost - 0 bcz we are solving a fesability problem 
    prob.solve()
    
    if prob.status == 'optimal':
        M = la.inv(W.value)
        print("Calculated Metric M:\n", M)
        return M
    else:
        print("System not contracting with these parameters.")
        return None

quad = PlanarQuadrotor()
bound = calculate_wind_bound(quad)
metric = calculate_metric(quad)