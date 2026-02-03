#!/usr/bin/env python3
"""
Alternative Approaches to IC-Independent Chaos Communication

This file contains experimental approaches explored before settling on the
IC header method used in chaos_rf.py. Each approach attempts to solve the
initial condition (IC) synchronization problem differently.

Approaches:
    1. Preamble: Send unmasked chaos before data for observer sync
    2. Key-derived ICs: Hash shared secret to deterministic ICs
    3. Canonical point: Start from known attractor event (x crosses threshold)
    4. IC header: Transmit quantized ICs (WINNER - used in chaos_rf.py)
    5. Adaptive observer: Error injection during preamble

Usage:
    python approaches.py 1          # Test approach 1 (preamble)
    python approaches.py 2          # Test approach 2 (key-derived)
    python approaches.py 3          # Test approach 3 (canonical)
    python approaches.py 4          # Test approach 4 (IC header)
    python approaches.py 5          # Test approach 5 (adaptive)
    python approaches.py all        # Test all approaches
"""

import numpy as np
import hashlib
from scipy.integrate import odeint

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.005
PERIOD_MIN, PERIOD_MAX = 500, 2000
X_MIN, X_MAX = -20.0, 20.0
SAMPLES_PER_BIT = 50
MASK_AMPLITUDE = 3.0

# IC header parameters
IC_BITS = 8
IC_HEADER_LEN = IC_BITS * 3
IC_X_RANGE = (-20, 20)
IC_Y_RANGE = (-30, 30)
IC_Z_RANGE = (0, 50)


# =============================================================================
# SHARED FUNCTIONS
# =============================================================================

def lorenz(state, t):
    x, y, z = state
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def generate_trajectory(n_samples, initial=None):
    if initial is None:
        initial = [1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1]
    t = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, initial, t)
    return traj[:n_samples]


def cuomo_oppenheim_observer(s_received, dt, x0=1.0, y0=1.0, z0=1.0):
    """Standard Cuomo-Oppenheim observer."""
    n = len(s_received)
    x_est, y_est, z_est = np.zeros(n), np.zeros(n), np.zeros(n)
    x_est[0], y_est[0], z_est[0] = x0, y0, z0

    for i in range(1, n):
        s = s_received[i - 1]
        x, y, z = x_est[i - 1], y_est[i - 1], z_est[i - 1]
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
            bits.append((ord(char) >> (7 - i)) & 1)
    return np.array(bits)


def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        if 32 <= byte < 127:
            chars.append(chr(byte))
    return ''.join(chars)


def quantize_ic(value, ic_range):
    lo, hi = ic_range
    clipped = np.clip(value, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    return int(norm * 255)


def dequantize_ic(quant, ic_range):
    lo, hi = ic_range
    return lo + (quant / 255) * (hi - lo)


def decode_bits_standard(residual, bits_tx, settle_frac=0.80, threshold_frac=0.45,
                         post_process=True, preamble_len=0):
    """Standard bit decoding with settle time and post-processing."""
    settle = int(SAMPLES_PER_BIT * settle_frac)
    threshold = MASK_AMPLITUDE * threshold_frac
    raw_residuals = []
    bits_rx = []

    for i in range(len(bits_tx)):
        start = preamble_len + i * SAMPLES_PER_BIT + settle
        end = preamble_len + (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            med = np.median(seg)
            raw_residuals.append(med)
            bits_rx.append(1 if med > threshold else 0)

    if post_process:
        for i in range(1, len(bits_rx)):
            if bits_rx[i - 1] == 1 and bits_rx[i] == 1:
                if raw_residuals[i] < MASK_AMPLITUDE * 1.2:
                    bits_rx[i] = 0

    return np.array(bits_rx), raw_residuals


def print_results(bits_tx, bits_rx, raw_residuals, preamble_len=0):
    """Print decode results."""
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

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


# =============================================================================
# APPROACH 1: PREAMBLE
# =============================================================================

def test_approach1_preamble(message="A", preamble_len=100):
    """
    Approach 1: Longer Unmasked Preamble

    Send unmasked chaos before data. Receiver uses arbitrary ICs,
    syncs during preamble, decodes after.
    """
    print("=" * 60)
    print(f"APPROACH 1: PREAMBLE ({preamble_len} samples)")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_data = len(bits_tx) * SAMPLES_PER_BIT
    n_total = preamble_len + n_data

    print(f"Message: \"{message}\" = {list(bits_tx)}")
    print(f"Preamble: {preamble_len}, Data: {n_data}, Total: {n_total}")

    # Generate trajectory with warmup
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

    # Simulate period encoding
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # Run observer with ARBITRARY ICs
    x_est, _, _ = cuomo_oppenheim_observer(s_recv, DT, 1.0, 1.0, 1.0)

    # Check sync at preamble end
    x_corr = np.corrcoef(x_true[preamble_len - 20:preamble_len],
                         x_est[preamble_len - 20:preamble_len])[0, 1]
    print(f"\nSync at preamble end: {x_corr:.4f}")

    residual = s_recv - x_est
    bits_rx, raw_residuals = decode_bits_standard(
        residual, bits_tx, settle_frac=0.5, threshold_frac=0.5,
        post_process=False, preamble_len=preamble_len
    )

    return print_results(bits_tx, bits_rx, raw_residuals, preamble_len)


# =============================================================================
# APPROACH 2: KEY-DERIVED ICS
# =============================================================================

def key_to_ic(key: str, warmup: int = 1000):
    """Derive ICs from shared key via hash."""
    h = hashlib.sha256(key.encode()).digest()
    x_raw = int.from_bytes(h[0:4], 'big') / (2 ** 32)
    y_raw = int.from_bytes(h[4:8], 'big') / (2 ** 32)
    z_raw = int.from_bytes(h[8:12], 'big') / (2 ** 32)
    x0 = -20 + x_raw * 40
    y0 = -30 + y_raw * 60
    z0 = z_raw * 50
    t = np.arange(0, warmup * DT, DT)
    traj = odeint(lorenz, [x0, y0, z0], t)
    return traj[-1, 0], traj[-1, 1], traj[-1, 2]


def test_approach2_key_ic(message="A", key="shared_secret_key"):
    """
    Approach 2: Key-Derived Initial Conditions

    Both TX and RX derive ICs from a shared secret key.
    key -> hash -> (x0, y0, z0) on the attractor
    """
    print("=" * 60)
    print(f"APPROACH 2: KEY-DERIVED ICs")
    print(f"  key=\"{key}\"")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")

    # Both derive ICs from key
    x0, y0, z0 = key_to_ic(key)
    print(f"Key-derived ICs: x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}")

    # TX generates trajectory from derived ICs
    t = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, [x0, y0, z0], t)[:n_samples]
    x_true = traj[:, 0]

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
    x_est, _, _ = cuomo_oppenheim_observer(s_recv, DT, x0, y0, z0)

    x_corr = np.corrcoef(x_true, x_est)[0, 1]
    print(f"\nx tracking correlation: {x_corr:.6f}")

    residual = s_recv - x_est
    bits_rx, raw_residuals = decode_bits_standard(residual, bits_tx)

    return print_results(bits_tx, bits_rx, raw_residuals)


# =============================================================================
# APPROACH 3: CANONICAL ATTRACTOR POINT
# =============================================================================

def find_canonical_event(traj, x_threshold=10.0, direction='up'):
    """Find index where x crosses threshold going up."""
    x = traj[:, 0]
    for i in range(1, len(x)):
        if direction == 'up':
            if x[i - 1] < x_threshold and x[i] >= x_threshold:
                return i
        else:
            if x[i - 1] > x_threshold and x[i] <= x_threshold:
                return i
    return None


def test_approach3_canonical(message="A", seed=42):
    """
    Approach 3: Canonical Attractor Point

    Both TX and RX agree to start from a canonical "marker event"
    -- when x crosses +10 going upward.
    """
    print("=" * 60)
    print(f"APPROACH 3: CANONICAL ATTRACTOR POINT")
    print(f"  seed={seed}, canonical x=10.0")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")

    np.random.seed(seed)

    # Find canonical event
    initial = [np.random.randn() * 10, np.random.randn() * 10, 25 + np.random.randn() * 5]
    warmup_steps = 5000
    t = np.arange(0, warmup_steps * DT, DT)
    warmup_traj = odeint(lorenz, initial, t)

    event_idx = find_canonical_event(warmup_traj)
    if event_idx is None:
        print("ERROR: No canonical event found")
        return False

    canonical_state = warmup_traj[event_idx]
    print(f"Canonical state: x={canonical_state[0]:.4f}, y={canonical_state[1]:.4f}, z={canonical_state[2]:.4f}")

    # Generate from canonical point
    t2 = np.arange(0, n_samples * DT, DT)
    traj = odeint(lorenz, canonical_state, t2)[:n_samples]
    x_true = traj[:, 0]

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Simulate encoding
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # RX also uses canonical state (same seed)
    x0, y0, z0 = canonical_state
    x_est, _, _ = cuomo_oppenheim_observer(s_recv, DT, x0, y0, z0)

    x_corr = np.corrcoef(x_true, x_est)[0, 1]
    print(f"\nx tracking correlation: {x_corr:.6f}")

    residual = s_recv - x_est
    bits_rx, raw_residuals = decode_bits_standard(
        residual, bits_tx, settle_frac=0.5, threshold_frac=0.5, post_process=False
    )

    return print_results(bits_tx, bits_rx, raw_residuals)


# =============================================================================
# APPROACH 4: IC HEADER (WINNER)
# =============================================================================

def ic_to_periods(x0, y0, z0):
    """Convert ICs to 24-period header."""
    x_quant = quantize_ic(x0, IC_X_RANGE)
    y_quant = quantize_ic(y0, IC_Y_RANGE)
    z_quant = quantize_ic(z0, IC_Z_RANGE)

    periods = []
    for quant in [x_quant, y_quant, z_quant]:
        for i in range(7, -1, -1):
            bit = (quant >> i) & 1
            periods.append(PERIOD_MIN if bit == 1 else PERIOD_MAX)

    return np.array(periods)


def periods_to_ic(periods):
    """Decode 24-period header to ICs."""
    quants = []
    for ic_idx in range(3):
        quant = 0
        for i in range(8):
            p = periods[ic_idx * 8 + i]
            mid = (PERIOD_MIN + PERIOD_MAX) / 2
            bit = 1 if p < mid else 0
            quant = (quant << 1) | bit
        quants.append(quant)

    x0 = dequantize_ic(quants[0], IC_X_RANGE)
    y0 = dequantize_ic(quants[1], IC_Y_RANGE)
    z0 = dequantize_ic(quants[2], IC_Z_RANGE)

    return x0, y0, z0


def test_approach4_ic_header(message="A"):
    """
    Approach 4: Transmit ICs in Header

    Prepend quantized x0, y0, z0 to the signal.
    Overhead: 24 periods for 8-bit precision per IC.
    THIS IS THE WINNING APPROACH used in chaos_rf.py.
    """
    print("=" * 60)
    print(f"APPROACH 4: IC HEADER (24-period overhead)")
    print("  *** THIS IS THE WINNING APPROACH ***")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")
    print(f"Data samples: {n_samples}")
    print(f"Header periods: {IC_HEADER_LEN}")

    # Generate trajectory with warmup
    warmup = 1000
    traj = generate_trajectory(warmup + n_samples)
    x_true = traj[warmup:warmup + n_samples, 0]
    y_true = traj[warmup:warmup + n_samples, 1]
    z_true = traj[warmup:warmup + n_samples, 2]

    x0, y0, z0 = x_true[0], y_true[0], z_true[0]
    print(f"\nTrue ICs: x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}")

    # Encode header
    header_periods = ic_to_periods(x0, y0, z0)

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Encode data
    data_periods = np.array([x_to_period(x) for x in s_tx])
    all_periods = np.concatenate([header_periods, data_periods])

    print(f"Total periods: {len(all_periods)}")

    # Simulate transmission
    periods_rx = all_periods

    # Decode header
    x0_rx, y0_rx, z0_rx = periods_to_ic(periods_rx[:IC_HEADER_LEN])
    print(f"Decoded ICs: x0={x0_rx:.4f}, y0={y0_rx:.4f}, z0={z0_rx:.4f}")
    print(f"IC error: dx={x0_rx - x0:.3f}, dy={y0_rx - y0:.3f}, dz={z0_rx - z0:.3f}")

    # Decode data
    data_periods_rx = periods_rx[IC_HEADER_LEN:]
    s_recv = np.array([period_to_x(p) for p in data_periods_rx])

    # Run observer with DECODED ICs
    x_est, _, _ = cuomo_oppenheim_observer(s_recv, DT, x0_rx, y0_rx, z0_rx)

    x_corr = np.corrcoef(x_true, x_est)[0, 1]
    print(f"\nx tracking correlation: {x_corr:.6f}")

    residual = s_recv - x_est
    bits_rx, raw_residuals = decode_bits_standard(residual, bits_tx)

    return print_results(bits_tx, bits_rx, raw_residuals)


# =============================================================================
# APPROACH 5: ADAPTIVE OBSERVER
# =============================================================================

def adaptive_observer(s_received, dt, preamble_len, k_sync=5.0, k_data=0.0,
                      x0=1.0, y0=1.0, z0=1.0):
    """
    Adaptive observer with variable error injection.

    dx/dt = sigma(y - x) + k*(s - x)
    k = k_sync during preamble, k = k_data during data
    """
    n = len(s_received)
    x_est, y_est, z_est = np.zeros(n), np.zeros(n), np.zeros(n)
    x_est[0], y_est[0], z_est[0] = x0, y0, z0

    for i in range(1, n):
        s = s_received[i - 1]
        x, y, z = x_est[i - 1], y_est[i - 1], z_est[i - 1]

        k = k_sync if i < preamble_len else k_data

        dx = SIGMA * (y - x) + k * (s - x)
        dy = s * (RHO - z) - y
        dz = s * y - BETA * z

        x_est[i] = x + dx * dt
        y_est[i] = y + dy * dt
        z_est[i] = z + dz * dt

    return x_est, y_est, z_est


def test_approach5_adaptive(message="A", preamble_len=100, k_sync=5.0, k_data=0.0):
    """
    Approach 5: Adaptive Observer with Error Injection

    Standard observer has dx/dt = sigma(y - x) -- x isn't coupled to s.
    Fix: Add error injection k*(s - x) during preamble to sync x.
    """
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

    # Create masked signal
    mask = np.zeros(n_total)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            start = preamble_len + i * SAMPLES_PER_BIT
            end = preamble_len + (i + 1) * SAMPLES_PER_BIT
            mask[start:end] = MASK_AMPLITUDE

    s_tx = x_true + mask

    # Simulate encoding
    periods = np.array([x_to_period(x) for x in s_tx])
    s_recv = np.array([period_to_x(p) for p in periods])

    # Run adaptive observer with ARBITRARY ICs
    x_est, _, _ = adaptive_observer(
        s_recv, DT, preamble_len, k_sync, k_data, 1.0, 1.0, 1.0
    )

    # Check sync at preamble end
    sync_start = max(0, preamble_len - 20)
    x_corr = np.corrcoef(x_true[sync_start:preamble_len],
                         x_est[sync_start:preamble_len])[0, 1]
    print(f"\nSync at preamble end: corr={x_corr:.4f}")

    residual = s_recv - x_est
    bits_rx, raw_residuals = decode_bits_standard(
        residual, bits_tx, settle_frac=0.5, threshold_frac=0.5,
        post_process=False, preamble_len=preamble_len
    )

    return print_results(bits_tx, bits_rx, raw_residuals, preamble_len)


# =============================================================================
# MAIN
# =============================================================================

def test_all(message="A", trials=5):
    """Run all approaches and compare."""
    print("=" * 60)
    print(f"TESTING ALL APPROACHES ({trials} trials each)")
    print("=" * 60)

    approaches = [
        ("1. Preamble", lambda: test_approach1_preamble(message)),
        ("2. Key-derived ICs", lambda: test_approach2_key_ic(message)),
        ("3. Canonical point", lambda: test_approach3_canonical(message)),
        ("4. IC header", lambda: test_approach4_ic_header(message)),
        ("5. Adaptive observer", lambda: test_approach5_adaptive(message)),
    ]

    results = {}
    for name, test_fn in approaches:
        successes = 0
        for _ in range(trials):
            if test_fn():
                successes += 1
        results[name] = successes / trials
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        bar = "#" * int(rate * 20)
        print(f"  {name:25s}: {rate * 100:5.1f}% {bar}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "1":
        test_approach1_preamble()
    elif cmd == "2":
        test_approach2_key_ic()
    elif cmd == "3":
        test_approach3_canonical()
    elif cmd == "4":
        test_approach4_ic_header()
    elif cmd == "5":
        test_approach5_adaptive()
    elif cmd == "all":
        test_all()
    else:
        print(f"Unknown approach: {cmd}")
        print("Use: 1, 2, 3, 4, 5, or all")
