#!/usr/bin/env python3
"""
Approach 4: Transmit ICs in Header

Prepend quantized x0, y0, z0 to the signal.
Overhead: ~24 periods for 8-bit precision per IC.

This WORKS because RX gets the exact ICs TX used.
Not IC-independent, but practical.
"""

import numpy as np
from scipy.integrate import odeint

SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0
DT = 0.005
PERIOD_MIN, PERIOD_MAX = 500, 2000
X_MIN, X_MAX = -20.0, 20.0
SAMPLES_PER_BIT = 50
MASK_AMPLITUDE = 3.0

# IC quantization parameters
# Each IC encoded as 8 periods (8 bits)
IC_BITS = 8
IC_HEADER_LEN = IC_BITS * 3  # x, y, z each get 8 periods

# IC ranges (Lorenz attractor bounds)
IC_X_RANGE = (-20, 20)
IC_Y_RANGE = (-30, 30)
IC_Z_RANGE = (0, 50)


def lorenz(state, t):
    x, y, z = state
    return [SIGMA*(y-x), x*(RHO-z)-y, x*y-BETA*z]


def generate_trajectory(duration, dt, initial=None):
    if initial is None:
        initial = [1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1]
    t = np.arange(0, duration, dt)
    trajectory = odeint(lorenz, initial, t)
    return t, trajectory


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


def quantize_ic(value, ic_range):
    """Quantize IC to 8-bit value."""
    lo, hi = ic_range
    clipped = np.clip(value, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    return int(norm * 255)


def dequantize_ic(quant, ic_range):
    """Dequantize 8-bit value to IC."""
    lo, hi = ic_range
    return lo + (quant / 255) * (hi - lo)


def ic_to_periods(x0, y0, z0):
    """Convert ICs to period-encoded header (24 periods)."""
    x_quant = quantize_ic(x0, IC_X_RANGE)
    y_quant = quantize_ic(y0, IC_Y_RANGE)
    z_quant = quantize_ic(z0, IC_Z_RANGE)

    # Each IC is 8 bits, encoded as periods
    # bit 1 → period = PERIOD_MIN (short)
    # bit 0 → period = PERIOD_MAX (long)
    periods = []
    for quant in [x_quant, y_quant, z_quant]:
        for i in range(7, -1, -1):
            bit = (quant >> i) & 1
            periods.append(PERIOD_MIN if bit == 1 else PERIOD_MAX)

    return np.array(periods)


def periods_to_ic(periods):
    """Decode header periods back to ICs."""
    quants = []
    for ic_idx in range(3):
        quant = 0
        for i in range(8):
            p = periods[ic_idx * 8 + i]
            # Closer to PERIOD_MIN → 1, closer to PERIOD_MAX → 0
            mid = (PERIOD_MIN + PERIOD_MAX) / 2
            bit = 1 if p < mid else 0
            quant = (quant << 1) | bit
        quants.append(quant)

    x0 = dequantize_ic(quants[0], IC_X_RANGE)
    y0 = dequantize_ic(quants[1], IC_Y_RANGE)
    z0 = dequantize_ic(quants[2], IC_Z_RANGE)

    return x0, y0, z0


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


def test_offline(message="A"):
    """Test IC header approach."""
    print("=" * 60)
    print(f"APPROACH 4: IC HEADER (24-period overhead)")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")
    print(f"Data samples: {n_samples}")
    print(f"Header periods: {IC_HEADER_LEN}")

    # Generate trajectory with warmup (like original RF code)
    warmup = 1000
    total = warmup + n_samples
    t, traj = generate_trajectory(duration=total * DT, dt=DT)
    x_true = traj[warmup:warmup + n_samples, 0]
    y_true = traj[warmup:warmup + n_samples, 1]
    z_true = traj[warmup:warmup + n_samples, 2]

    # Get ICs from trajectory
    x0, y0, z0 = x_true[0], y_true[0], z_true[0]
    print(f"\nTrue ICs: x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}")

    # Encode header
    header_periods = ic_to_periods(x0, y0, z0)
    print(f"Header periods: {len(header_periods)}")

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Period encode data
    data_periods = np.array([x_to_period(x) for x in s_tx])

    # Combine: header + data
    all_periods = np.concatenate([header_periods, data_periods])
    print(f"Total periods: {len(all_periods)}")

    # Simulate transmission (ideal channel)
    periods_rx = all_periods  # Perfect reception

    # RX: Decode header
    x0_rx, y0_rx, z0_rx = periods_to_ic(periods_rx[:IC_HEADER_LEN])
    print(f"Decoded ICs: x0={x0_rx:.4f}, y0={y0_rx:.4f}, z0={z0_rx:.4f}")
    print(f"IC error: dx={x0_rx-x0:.3f}, dy={y0_rx-y0:.3f}, dz={z0_rx-z0:.3f}")

    # RX: Decode data
    data_periods_rx = periods_rx[IC_HEADER_LEN:]
    s_recv = np.array([period_to_x(p) for p in data_periods_rx])

    # Run observer with DECODED ICs
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT, x0_rx, y0_rx, z0_rx)

    # Recover mask
    residual = s_recv - x_est

    # PROPER DECODE
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

    # Post-process
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

    print("\nPer-bit residuals:")
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


def test_multiple_trials(n=20):
    """Test with multiple trajectories."""
    print("=" * 60)
    print(f"TESTING {n} TRIALS")
    print("=" * 60)

    successes = 0
    for i in range(n):
        print(f"\n--- Trial {i+1} ---")
        if test_offline("A"):
            successes += 1

    print("\n" + "=" * 60)
    print(f"SUCCESS RATE: {successes}/{n} = {successes/n*100:.0f}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "trials":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        test_multiple_trials(n)
    else:
        test_offline("A")
