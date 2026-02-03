#!/usr/bin/env python3
"""
Approach 1: Longer Unmasked Preamble

Solve IC dependency by sending unmasked chaos before data.
Receiver uses arbitrary ICs, syncs during preamble, decodes after.
"""

import numpy as np
from scipy.integrate import odeint
from scipy.ndimage import uniform_filter1d

# Lorenz parameters
SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0

# Encoding parameters
DT = 0.005
PERIOD_MIN, PERIOD_MAX = 500, 2000
X_MIN, X_MAX = -20.0, 20.0

# Modem parameters
SAMPLES_PER_BIT = 50
MASK_AMPLITUDE = 3.0
PREAMBLE_LENGTH = 100  # Unmasked samples before data


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


def cuomo_oppenheim_observer(s_received, dt, x0=1.0, y0=1.0, z0=1.0):
    """Observer with arbitrary ICs - the point of this approach."""
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


def test_offline(message="A", preamble_len=100):
    """Test preamble approach without RF."""
    print("=" * 60)
    print(f"APPROACH 1: PREAMBLE ({preamble_len} samples)")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_data = len(bits_tx) * SAMPLES_PER_BIT
    n_total = preamble_len + n_data

    print(f"Message: \"{message}\" = {list(bits_tx)}")
    print(f"Preamble: {preamble_len}, Data: {n_data}, Total: {n_total}")

    # Generate trajectory (with warmup to reach attractor)
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

    # Run observer with ARBITRARY ICs (1, 1, 1) - the whole point
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT, 1.0, 1.0, 1.0)

    # Recover mask
    residual = s_recv - x_est

    # Decode bits from data portion (after preamble)
    settle = int(SAMPLES_PER_BIT * 0.5)
    threshold = MASK_AMPLITUDE * 0.5
    bits_rx = []

    for i in range(len(bits_tx)):
        start = preamble_len + i * SAMPLES_PER_BIT + settle
        end = preamble_len + (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            bits_rx.append(1 if np.median(seg) > threshold else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

    # Check sync quality at end of preamble
    preamble_end = preamble_len
    x_corr_preamble = np.corrcoef(x_true[preamble_end-20:preamble_end],
                                   x_est[preamble_end-20:preamble_end])[0, 1]

    print(f"\nSync quality at preamble end: {x_corr_preamble:.4f}")
    print(f"\nTX: {list(bits_tx)}")
    print(f"RX: {list(bits_rx)}")
    print(f"Errors: {errors}/{len(bits_tx)}")

    # Per-bit analysis
    print("\nPer-bit residuals:")
    for i in range(min(len(bits_tx), len(bits_rx))):
        start = preamble_len + i * SAMPLES_PER_BIT + settle
        end = preamble_len + (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            expected = MASK_AMPLITUDE if bits_tx[i] == 1 else 0
            status = "OK" if bits_tx[i] == bits_rx[i] else "ERR"
            print(f"  bit {i}: tx={bits_tx[i]} rx={bits_rx[i]} "
                  f"residual={np.median(seg):+.2f} expected={expected:.0f} [{status}]")

    if errors == 0:
        print(f"\n*** SUCCESS: \"{bits_to_text(bits_rx)}\" ***")
    else:
        print(f"\n*** FAILED: {errors} errors ***")

    return errors == 0


def sweep_preamble_lengths(message="A"):
    """Test various preamble lengths."""
    print("=" * 60)
    print("PREAMBLE LENGTH SWEEP")
    print("=" * 60)

    results = []
    for preamble_len in [25, 50, 75, 100, 150, 200, 300]:
        # Run multiple trials
        successes = 0
        trials = 10
        for _ in range(trials):
            if test_offline(message, preamble_len):
                successes += 1
        rate = successes / trials
        results.append((preamble_len, rate))
        print(f"\nPreamble {preamble_len}: {successes}/{trials} = {rate*100:.0f}%\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for preamble_len, rate in results:
        bar = "#" * int(rate * 20)
        print(f"  {preamble_len:3d} samples: {rate*100:5.1f}% {bar}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep_preamble_lengths()
    else:
        # Single test with default 100-sample preamble
        test_offline("A", 100)
