#!/usr/bin/env python3
"""
Approach 2 v2: Key-Derived ICs with CORRECT decode parameters

Uses the same decode logic that achieves 100% in the RF test:
- 80% settle time
- 0.45 threshold
- Post-processing for 1→0 transients
"""

import numpy as np
import hashlib
from scipy.integrate import odeint

SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0
DT = 0.005
PERIOD_MIN, PERIOD_MAX = 500, 2000
X_MIN, X_MAX = -20.0, 20.0
SAMPLES_PER_BIT = 50
MASK_AMPLITUDE = 3.0


def lorenz(state, t):
    x, y, z = state
    return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]


def key_to_ic(key: str, warmup: int = 1000):
    """Derive ICs from shared key."""
    h = hashlib.sha256(key.encode()).digest()
    x_raw = int.from_bytes(h[0:4], 'big') / (2**32)
    y_raw = int.from_bytes(h[4:8], 'big') / (2**32)
    z_raw = int.from_bytes(h[8:12], 'big') / (2**32)
    x0 = -20 + x_raw * 40
    y0 = -30 + y_raw * 60
    z0 = z_raw * 50
    t = np.arange(0, warmup * DT, DT)
    traj = odeint(lorenz, [x0, y0, z0], t)
    return traj[-1, 0], traj[-1, 1], traj[-1, 2]


def generate_trajectory(n_samples, x0, y0, z0):
    t = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, [x0, y0, z0], t)
    return traj[:n_samples]


def cuomo_oppenheim_observer(s_received, dt, x0, y0, z0):
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


def test_offline(message="A", key="shared_secret_key"):
    """Test with proper RF decode parameters."""
    print("=" * 60)
    print(f"APPROACH 2 v2: KEY-DERIVED ICs (correct decode)")
    print(f"  key=\"{key}\"")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")

    # Both derive ICs from key
    x0, y0, z0 = key_to_ic(key)
    print(f"Key-derived ICs: x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}")

    # TX generates trajectory
    traj = generate_trajectory(n_samples, x0, y0, z0)
    x_true = traj[:, 0]
    y_true = traj[:, 1]
    z_true = traj[:, 2]

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Simulate encoding
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # RX derives SAME ICs from key
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT, x0, y0, z0)

    # Recover mask
    residual = s_recv - x_est

    # PROPER DECODE: 80% settle, 0.45 threshold, post-processing
    settle = int(SAMPLES_PER_BIT * 0.80)
    threshold = MASK_AMPLITUDE * 0.45
    raw_residuals = []
    bits_rx = []

    for i in range(len(bits_tx)):
        start = i * SAMPLES_PER_BIT + settle
        end = (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            med = np.median(seg)
            raw_residuals.append(med)
            bits_rx.append(1 if med > threshold else 0)

    # Post-process: correct 1→0 transients
    for i in range(1, len(bits_rx)):
        if bits_rx[i-1] == 1 and bits_rx[i] == 1:
            if raw_residuals[i] < MASK_AMPLITUDE * 1.2:
                bits_rx[i] = 0

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

    x_corr = np.corrcoef(x_true, x_est)[0, 1]
    print(f"\nx tracking correlation: {x_corr:.6f}")

    print(f"\nTX: {list(bits_tx)}")
    print(f"RX: {list(bits_rx)}")
    print(f"Errors: {errors}/{len(bits_tx)}")

    print("\nPer-bit residuals (80% settle):")
    for i in range(min(len(bits_tx), len(bits_rx))):
        expected = MASK_AMPLITUDE if bits_tx[i] == 1 else 0
        status = "OK" if bits_tx[i] == bits_rx[i] else "ERR"
        print(f"  bit {i}: tx={bits_tx[i]} rx={bits_rx[i]} "
              f"residual={raw_residuals[i]:+.2f} expected={expected:.0f} [{status}]")

    if errors == 0:
        print(f"\n*** SUCCESS ***")
    else:
        print(f"\n*** FAILED: {errors} errors ***")

    return errors == 0


def test_multiple_keys(message="A", n_keys=20):
    """Test with multiple random keys."""
    print("=" * 60)
    print(f"TESTING {n_keys} RANDOM KEYS")
    print("=" * 60)

    successes = 0
    for i in range(n_keys):
        key = f"test_key_{i}_{np.random.randint(10000)}"
        if test_offline(message, key):
            successes += 1
        print()

    print("=" * 60)
    print(f"SUCCESS RATE: {successes}/{n_keys} = {successes/n_keys*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "keys":
        test_multiple_keys()
    else:
        test_offline("A")
