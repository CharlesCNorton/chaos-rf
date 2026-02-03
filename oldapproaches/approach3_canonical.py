#!/usr/bin/env python3
"""
Approach 3: Canonical Attractor Point

Lorenz attractor has a well-known trajectory structure.
Both TX and RX agree to start from a canonical "marker event"
— e.g., when x crosses +10 going upward.

The trajectory leading up to and after this event is deterministic.
"""

import numpy as np
from scipy.integrate import odeint

# Lorenz parameters
SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0

# Encoding parameters
DT = 0.005
PERIOD_MIN, PERIOD_MAX = 500, 2000
X_MIN, X_MAX = -20.0, 20.0

# Modem parameters
SAMPLES_PER_BIT = 50
MASK_AMPLITUDE = 3.0

# Canonical point: x crosses this value going up
CANONICAL_X = 10.0


def lorenz(state, t):
    x, y, z = state
    return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]


def find_canonical_event(traj, x_threshold=CANONICAL_X, direction='up'):
    """
    Find index where x crosses threshold going up (or down).
    Returns index of first such crossing, or None.
    """
    x = traj[:, 0]
    for i in range(1, len(x)):
        if direction == 'up':
            if x[i-1] < x_threshold and x[i] >= x_threshold:
                return i
        else:
            if x[i-1] > x_threshold and x[i] <= x_threshold:
                return i
    return None


def generate_from_canonical(n_samples, seed=None):
    """
    Generate trajectory starting from canonical event.

    1. Run Lorenz from arbitrary IC for warmup
    2. Find canonical event (x crosses CANONICAL_X going up)
    3. Return trajectory starting from that point

    Both TX and RX call this with same seed to get same trajectory.
    """
    if seed is not None:
        np.random.seed(seed)

    # Start from random IC, run long warmup to find canonical event
    initial = [np.random.randn() * 10, np.random.randn() * 10, 25 + np.random.randn() * 5]
    warmup_steps = 5000
    t = np.arange(0, warmup_steps * DT, DT)
    warmup_traj = odeint(lorenz, initial, t)

    # Find canonical event
    event_idx = find_canonical_event(warmup_traj)
    if event_idx is None:
        # Shouldn't happen with enough warmup
        raise RuntimeError("No canonical event found")

    # Get state at canonical event
    canonical_state = warmup_traj[event_idx]

    # Generate trajectory from canonical point
    t2 = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, canonical_state, t2)

    return traj[:n_samples], canonical_state


def pecora_carroll_sync(x_received, dt, y0, z0):
    """Standard Pecora-Carroll sync for y,z."""
    n = len(x_received)
    y_recv, z_recv = np.zeros(n), np.zeros(n)
    y_recv[0], z_recv[0] = y0, z0

    for i in range(1, n):
        x = x_received[i-1]
        y, z = y_recv[i-1], z_recv[i-1]
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        y_recv[i] = y + dy * dt
        z_recv[i] = z + dz * dt

    return y_recv, z_recv


def cuomo_oppenheim_observer(s_received, dt, x0, y0, z0):
    """Standard observer."""
    n = len(s_received)
    x_est, y_est, z_est = np.zeros(n), np.zeros(n), np.zeros(n)
    x_est[0], y_est[0], z_est[0] = x0, y0, z0

    for i in range(1, n):
        s = s_received[i-1]
        x, y, z = x_est[i-1], y_est[i-1], z_est[i-1]
        dx = SIGMA * (y - x)
        dy = s * (RHO - z) - y
        dz = s * y - BETA * z
        x_est[i] = x + dx * dt
        y_est[i] = y + dy * dt
        z_est[i] = z + dz * dt

    return x_est, y_est, z_est


def x_to_period(x):
    x_clipped = np.clip(x, X_MIN, X_MAX)
    x_norm = (x_clipped - X_MIN) / (X_MAX - X_MIN)
    return int(PERIOD_MAX - x_norm * (PERIOD_MAX - PERIOD_MIN))


def period_to_x(period):
    p_clipped = np.clip(period, PERIOD_MIN, PERIOD_MAX)
    x_norm = (PERIOD_MAX - p_clipped) / (PERIOD_MAX - PERIOD_MIN)
    return X_MIN + x_norm * (X_MAX - X_MIN)


def text_to_bits(text):
    bits = []
    for char in text:
        for i in range(8):
            bits.append((ord(char) >> (7-i)) & 1)
    return np.array(bits)


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits)-7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i+j])
        if 32 <= byte < 127:
            chars.append(chr(byte))
    return ''.join(chars)


def test_offline(message="A", seed=42):
    """Test canonical point approach without RF."""
    print("=" * 60)
    print(f"APPROACH 3: CANONICAL ATTRACTOR POINT")
    print(f"  seed={seed}, canonical x={CANONICAL_X}")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")

    # TX generates trajectory from canonical point
    traj, canonical_state = generate_from_canonical(n_samples, seed)
    x_true = traj[:, 0]
    y_true = traj[:, 1]
    z_true = traj[:, 2]

    print(f"Canonical state: x={canonical_state[0]:.4f}, y={canonical_state[1]:.4f}, z={canonical_state[2]:.4f}")

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            start = i * SAMPLES_PER_BIT
            end = (i + 1) * SAMPLES_PER_BIT
            mask[start:end] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Simulate period encoding (ideal channel)
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # RX also generates from canonical point (same seed)
    # In practice, seed would be pre-shared or derived from key
    traj_rx, canonical_state_rx = generate_from_canonical(n_samples, seed)

    # Verify TX and RX have same trajectory
    x_match = np.allclose(traj[:, 0], traj_rx[:, 0])
    print(f"TX/RX trajectory match: {x_match}")

    # RX runs observer with canonical ICs
    x0, y0, z0 = canonical_state_rx
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT, x0, y0, z0)

    # Recover mask
    residual = s_recv - x_est

    # Decode bits
    settle = int(SAMPLES_PER_BIT * 0.5)
    threshold = MASK_AMPLITUDE * 0.5
    bits_rx = []
    residuals = []

    for i in range(len(bits_tx)):
        start = i * SAMPLES_PER_BIT + settle
        end = (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            med = np.median(seg)
            residuals.append(med)
            bits_rx.append(1 if med > threshold else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

    # Check x tracking
    x_corr = np.corrcoef(x_true, x_est)[0, 1]
    print(f"\nx tracking correlation: {x_corr:.6f}")

    print(f"\nTX: {list(bits_tx)}")
    print(f"RX: {list(bits_rx)}")
    print(f"Errors: {errors}/{len(bits_tx)}")

    # Per-bit analysis
    print("\nPer-bit residuals:")
    for i in range(min(len(bits_tx), len(bits_rx))):
        expected = MASK_AMPLITUDE if bits_tx[i] == 1 else 0
        status = "OK" if bits_tx[i] == bits_rx[i] else "ERR"
        print(f"  bit {i}: tx={bits_tx[i]} rx={bits_rx[i]} "
              f"residual={residuals[i]:+.2f} expected={expected:.0f} [{status}]")

    if errors == 0:
        print(f"\n*** SUCCESS: \"{bits_to_text(bits_rx)}\" ***")
    else:
        print(f"\n*** FAILED: {errors} errors ***")

    return errors == 0


def test_multiple_seeds(message="A", n_seeds=10):
    """Test with multiple seeds."""
    print("=" * 60)
    print(f"TESTING {n_seeds} DIFFERENT SEEDS")
    print("=" * 60)

    successes = 0
    for seed in range(n_seeds):
        if test_offline(message, seed):
            successes += 1
        print()

    print("=" * 60)
    print(f"SUMMARY: {successes}/{n_seeds} = {successes/n_seeds*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        test_multiple_seeds()
    else:
        seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
        test_offline("A", seed)
