#!/usr/bin/env python3
"""
Approach 5: Adaptive Observer with Error Injection

Standard Cuomo-Oppenheim observer has dx/dt = σ(y - x) — x isn't coupled to s.
This causes x drift even after y,z sync.

Fix: Add error injection term k*(s - x) to dx equation:
    dx/dt = σ(y - x) + k*(s - x)

During preamble (unmasked): k > 0 syncs x to received signal
During data (masked): k → 0 so x follows natural dynamics, residual reveals mask
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
PREAMBLE_LENGTH = 100


def lorenz(state, t):
    x, y, z = state
    return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]


def generate_trajectory(n_samples, initial=None):
    if initial is None:
        initial = [1.0 + np.random.randn()*0.1,
                   1.0 + np.random.randn()*0.1,
                   1.0 + np.random.randn()*0.1]
    t = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, initial, t)
    return traj[:n_samples]


def adaptive_observer(s_received, dt, preamble_len, k_sync=5.0, k_data=0.0,
                      x0=1.0, y0=1.0, z0=1.0):
    """
    Adaptive Cuomo-Oppenheim observer with variable error injection.

    dx/dt = σ(y - x) + k*(s - x)
    dy/dt = s*(ρ - z) - y
    dz/dt = s*y - β*z

    k = k_sync during preamble (high, to sync x)
    k = k_data during data (low or zero, so x follows natural dynamics)
    """
    n = len(s_received)
    x_est, y_est, z_est = np.zeros(n), np.zeros(n), np.zeros(n)
    x_est[0], y_est[0], z_est[0] = x0, y0, z0

    for i in range(1, n):
        s = s_received[i-1]
        x, y, z = x_est[i-1], y_est[i-1], z_est[i-1]

        # Variable coupling strength
        k = k_sync if i < preamble_len else k_data

        # Modified observer with error injection
        dx = SIGMA * (y - x) + k * (s - x)
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


def test_offline(message="A", preamble_len=100, k_sync=5.0, k_data=0.0):
    """Test adaptive observer without RF."""
    print("=" * 60)
    print(f"APPROACH 5: ADAPTIVE OBSERVER")
    print(f"  preamble={preamble_len}, k_sync={k_sync}, k_data={k_data}")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_data = len(bits_tx) * SAMPLES_PER_BIT
    n_total = preamble_len + n_data

    print(f"Message: \"{message}\" = {list(bits_tx)}")

    # Generate trajectory
    warmup = 1000
    traj = generate_trajectory(warmup + n_total)
    x_true = traj[warmup:warmup + n_total, 0]

    # Create masked signal: preamble (unmasked) + data (masked)
    mask = np.zeros(n_total)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            start = preamble_len + i * SAMPLES_PER_BIT
            end = preamble_len + (i + 1) * SAMPLES_PER_BIT
            mask[start:end] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Simulate period encoding (ideal channel)
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # Run adaptive observer with ARBITRARY ICs
    x_est, y_est, z_est = adaptive_observer(
        s_recv, DT, preamble_len, k_sync, k_data, 1.0, 1.0, 1.0
    )

    # Check sync at end of preamble
    sync_start = max(0, preamble_len - 20)
    x_corr = np.corrcoef(x_true[sync_start:preamble_len],
                         x_est[sync_start:preamble_len])[0, 1]
    x_rmse = np.sqrt(np.mean((x_true[sync_start:preamble_len] -
                              x_est[sync_start:preamble_len])**2))
    print(f"\nSync at preamble end: corr={x_corr:.4f}, RMSE={x_rmse:.3f}")

    # Recover mask
    residual = s_recv - x_est

    # Decode bits from data portion
    settle = int(SAMPLES_PER_BIT * 0.5)
    threshold = MASK_AMPLITUDE * 0.5
    bits_rx = []
    residuals = []

    for i in range(len(bits_tx)):
        start = preamble_len + i * SAMPLES_PER_BIT + settle
        end = preamble_len + (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            med = np.median(seg)
            residuals.append(med)
            bits_rx.append(1 if med > threshold else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

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
        print(f"\n*** SUCCESS ***")
    else:
        print(f"\n*** FAILED: {errors} errors ***")

    return errors == 0


def sweep_parameters():
    """Sweep k_sync and k_data to find optimal values."""
    print("=" * 60)
    print("PARAMETER SWEEP")
    print("=" * 60)

    results = []

    for k_sync in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for k_data in [0.0, 0.1, 0.5, 1.0]:
            successes = 0
            trials = 10
            for _ in range(trials):
                if test_offline("A", 100, k_sync, k_data):
                    successes += 1
            rate = successes / trials
            results.append((k_sync, k_data, rate))
            print(f"\nk_sync={k_sync}, k_data={k_data}: {successes}/{trials} = {rate*100:.0f}%\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k_sync, k_data, rate in sorted(results, key=lambda x: -x[2]):
        bar = "#" * int(rate * 20)
        print(f"  k_sync={k_sync:5.1f}, k_data={k_data:4.1f}: {rate*100:5.1f}% {bar}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep_parameters()
    else:
        # Single test
        test_offline("A", 100, k_sync=5.0, k_data=0.0)
