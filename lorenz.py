"""Lorenz system integrator for chaotic synchronization experiment."""
import numpy as np
from scipy.integrate import odeint

# Standard Lorenz parameters
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

def lorenz(state, t, sigma=SIGMA, rho=RHO, beta=BETA):
    """Lorenz system derivatives."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def generate_trajectory(duration=1.0, dt=0.001, initial=None):
    """Generate Lorenz trajectory.
    
    Args:
        duration: Total time in Lorenz time units
        dt: Time step
        initial: Initial state [x0, y0, z0], random if None
    
    Returns:
        t: Time array
        trajectory: Array of shape (N, 3) with [x, y, z]
    """
    if initial is None:
        # Random initial condition on the attractor
        initial = [1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1]
    
    t = np.arange(0, duration, dt)
    trajectory = odeint(lorenz, initial, t)
    return t, trajectory

def lorenz_receiver(x_received, y_prev, z_prev, dt, sigma=SIGMA, rho=RHO, beta=BETA):
    """Single step of receiver Lorenz system (Pecora-Carroll).
    
    Substitutes received x into the dynamics for y and z.
    Uses simple Euler integration for real-time operation.
    """
    # dy/dt = x(rho - z) - y
    # dz/dt = xy - beta*z
    dy = x_received * (rho - z_prev) - y_prev
    dz = x_received * y_prev - beta * z_prev
    
    y_new = y_prev + dy * dt
    z_new = z_prev + dz * dt
    return y_new, z_new

if __name__ == "__main__":
    # Test: generate and print trajectory stats
    t, traj = generate_trajectory(duration=10.0, dt=0.001)
    x, y, z = traj.T
    print(f"Generated {len(t)} points over {t[-1]:.1f} time units")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    print(f"z range: [{z.min():.2f}, {z.max():.2f}]")
