"""Encode Lorenz x(t) with Flipper-compatible timing."""
import numpy as np
from lorenz import generate_trajectory

# Flipper-compatible timing (500µs base works well)
FREQ_HZ = 433920000
PERIOD_MIN_US = 500   # High x -> fast (500µs period = 2kHz)
PERIOD_MAX_US = 2000  # Low x -> slow (2ms period = 500Hz)
X_MIN = -20.0
X_MAX = 20.0

def x_to_period(x):
    x_norm = (np.clip(x, X_MIN, X_MAX) - X_MIN) / (X_MAX - X_MIN)
    return int(PERIOD_MAX_US - x_norm * (PERIOD_MAX_US - PERIOD_MIN_US))

def encode_trajectory(x_values):
    """Each x -> one pulse cycle with period encoding the value."""
    pulses = []
    for x in x_values:
        period = x_to_period(x)
        half = period // 2
        pulses.append(half)   # High
        pulses.append(-half)  # Low (negative for .sub format)
    return pulses

def write_sub_file(pulses, filename):
    with open(filename, 'w') as f:
        f.write("Filetype: Flipper SubGhz RAW File\n")
        f.write("Version: 1\n")
        f.write(f"Frequency: {FREQ_HZ}\n")
        f.write("Preset: FuriHalSubGhzPresetOok650Async\n")
        f.write("Protocol: RAW\n")
        
        chunk_size = 500
        for i in range(0, len(pulses), chunk_size):
            chunk = pulses[i:i+chunk_size]
            f.write("RAW_Data: " + " ".join(map(str, chunk)) + "\n")

def generate_chaos_signal(n_samples=100, dt=0.01, filename="chaos_v2.sub"):
    """Generate slower chaos signal for reliable Flipper TX."""
    t, traj = generate_trajectory(duration=n_samples * dt, dt=dt)
    x = traj[:n_samples, 0]
    y = traj[:n_samples, 1]
    z = traj[:n_samples, 2]
    
    pulses = encode_trajectory(x)
    write_sub_file(pulses, filename)
    
    # Calculate duration
    total_us = sum(abs(p) for p in pulses)
    
    return {
        'n_samples': len(x),
        'duration_sec': total_us / 1e6,
        'x': x, 'y': y, 'z': z,
        'filename': filename
    }

if __name__ == "__main__":
    info = generate_chaos_signal(n_samples=50, dt=0.02)
    print(f"Generated {info['filename']}")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Duration: {info['duration_sec']:.2f} sec")
    print(f"  x range: [{info['x'].min():.2f}, {info['x'].max():.2f}]")
    
    # Show first few periods
    pulses = encode_trajectory(info['x'][:5])
    print(f"  First 5 periods (µs): {[abs(p)*2 for p in pulses[::2]]}")
