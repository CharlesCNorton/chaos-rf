"""Receiver-side Lorenz system with Pecora-Carroll synchronization."""
import numpy as np
from lorenz import SIGMA, RHO, BETA, generate_trajectory

def pecora_carroll_sync(x_received, dt, y0=1.0, z0=1.0):
    """Run Pecora-Carroll synchronization.
    
    Given received x(t), integrate the receiver subsystem:
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
    
    Using received x instead of local x.
    
    Args:
        x_received: Array of received x values
        dt: Time step between samples (in Lorenz time units)
        y0, z0: Initial conditions for receiver
    
    Returns:
        y_recv: Receiver y trajectory
        z_recv: Receiver z trajectory
    """
    n = len(x_received)
    y_recv = np.zeros(n)
    z_recv = np.zeros(n)
    
    y_recv[0] = y0
    z_recv[0] = z0
    
    for i in range(1, n):
        x = x_received[i-1]
        y = y_recv[i-1]
        z = z_recv[i-1]
        
        # Euler integration of receiver subsystem
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        
        y_recv[i] = y + dy * dt
        z_recv[i] = z + dz * dt
    
    return y_recv, z_recv

def synchronization_error(y_true, z_true, y_recv, z_recv, skip_transient=100):
    """Compute synchronization error after transient.
    
    Returns:
        rms_y: RMS error in y
        rms_z: RMS error in z
        correlation_y: Correlation coefficient for y
        correlation_z: Correlation coefficient for z
    """
    # Skip initial transient
    y_t = y_true[skip_transient:]
    z_t = z_true[skip_transient:]
    y_r = y_recv[skip_transient:]
    z_r = z_recv[skip_transient:]
    
    # RMS error
    rms_y = np.sqrt(np.mean((y_t - y_r)**2))
    rms_z = np.sqrt(np.mean((z_t - z_r)**2))
    
    # Correlation
    corr_y = np.corrcoef(y_t, y_r)[0, 1]
    corr_z = np.corrcoef(z_t, z_r)[0, 1]
    
    return {
        "rms_y": rms_y,
        "rms_z": rms_z,
        "correlation_y": corr_y,
        "correlation_z": corr_z
    }

def test_ideal_sync():
    """Test synchronization with ideal (no-noise) x transmission."""
    print("Testing ideal synchronization (no RF, direct x)...")
    
    # Generate transmitter trajectory
    duration = 10.0
    dt = 0.002
    t, traj = generate_trajectory(duration=duration, dt=dt)
    x_true = traj[:, 0]
    y_true = traj[:, 1]
    z_true = traj[:, 2]
    
    # Run receiver with different initial conditions
    y_recv, z_recv = pecora_carroll_sync(x_true, dt, y0=10.0, z0=30.0)
    
    # Compute error
    err = synchronization_error(y_true, z_true, y_recv, z_recv)
    
    print(f"  Samples: {len(x_true)}")
    print(f"  RMS error y: {err['rms_y']:.6f}")
    print(f"  RMS error z: {err['rms_z']:.6f}")
    print(f"  Correlation y: {err['correlation_y']:.6f}")
    print(f"  Correlation z: {err['correlation_z']:.6f}")
    
    return err

if __name__ == "__main__":
    test_ideal_sync()
