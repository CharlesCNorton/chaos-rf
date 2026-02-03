#!/usr/bin/env python3
"""
Chaotic Synchronization and Communication over 433 MHz RF

Demonstrates Pecora-Carroll synchronization using a Flipper Zero transmitter
and RTL-SDR receiver on a Raspberry Pi 5.

Usage:
    python chaos_rf.py sync [n_samples]     # Synchronization experiment
    python chaos_rf.py modem [message]      # Chaos-masked communication
    python chaos_rf.py test                 # Offline modem test (no RF)

References:
    - Pecora & Carroll, "Synchronization in Chaotic Systems", PRL 1990
    - Cuomo & Oppenheim, "Circuit Implementation of Synchronized Chaos", PRL 1993
"""

import subprocess
import serial
import time
import numpy as np
from scipy.integrate import odeint
from scipy.ndimage import uniform_filter1d

# =============================================================================
# LORENZ SYSTEM
# =============================================================================

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0


def lorenz(state, t):
    """Lorenz system derivatives."""
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z
    ]


def generate_trajectory(duration, dt, initial=None):
    """Generate Lorenz trajectory.

    Args:
        duration: Total time in Lorenz time units
        dt: Integration timestep
        initial: Initial state [x0, y0, z0], defaults to [1,1,1] + noise

    Returns:
        t: Time array
        trajectory: Array of shape (N, 3) with [x, y, z]
    """
    if initial is None:
        initial = [1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1,
                   1.0 + np.random.randn() * 0.1]
    t = np.arange(0, duration, dt)
    trajectory = odeint(lorenz, initial, t)
    return t, trajectory


# =============================================================================
# SYNCHRONIZATION
# =============================================================================

def pecora_carroll_sync(x_received, dt, y0=1.0, z0=1.0):
    """Pecora-Carroll synchronization.

    Receiver integrates y,z subsystem driven by received x:
        dy/dt = x(rho - z) - y
        dz/dt = xy - beta*z

    After transient, receiver's y,z converge to transmitter's y,z.
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

        dy = x * (RHO - z) - y
        dz = x * y - BETA * z

        y_recv[i] = y + dy * dt
        z_recv[i] = z + dz * dt

    return y_recv, z_recv


def cuomo_oppenheim_observer(s_received, dt, x0=1.0, y0=1.0, z0=1.0):
    """Cuomo-Oppenheim observer for chaos-masked signals.

    For masked signal s = x + m:
        - y,z subsystem is driven by s (tracks perturbed trajectory)
        - x subsystem evolves via natural dynamics (low-pass filters mask)

    Observer equations:
        dx_hat/dt = sigma*(y_hat - x_hat)    <- filters mask
        dy_hat/dt = s*(rho - z_hat) - y_hat  <- driven by s
        dz_hat/dt = s*y_hat - beta*z_hat     <- driven by s

    Returns x_est, y_est, z_est
    """
    n = len(s_received)
    x_est = np.zeros(n)
    y_est = np.zeros(n)
    z_est = np.zeros(n)
    x_est[0] = x0
    y_est[0] = y0
    z_est[0] = z0

    for i in range(1, n):
        s = s_received[i-1]
        x = x_est[i-1]
        y = y_est[i-1]
        z = z_est[i-1]

        # x evolves via natural dynamics (NO coupling to s)
        dx = SIGMA * (y - x)
        # y,z driven by received signal s
        dy = s * (RHO - z) - y
        dz = s * y - BETA * z

        x_est[i] = x + dx * dt
        y_est[i] = y + dy * dt
        z_est[i] = z + dz * dt

    return x_est, y_est, z_est


# =============================================================================
# RF ENCODING
# =============================================================================

# Encoding parameters
DT = 0.005              # Lorenz integration timestep
PERIOD_MIN = 500        # Minimum pulse period (microseconds)
PERIOD_MAX = 2000       # Maximum pulse period (microseconds)
X_MIN, X_MAX = -20.0, 20.0

# Modem parameters
SAMPLES_PER_BIT = 150  # Reduced for faster upload (150 causes serial hang)
MASK_AMPLITUDE = 3.0


def x_to_period(x):
    """Map Lorenz x value to pulse period in microseconds."""
    x_clipped = np.clip(x, X_MIN, X_MAX)
    x_norm = (x_clipped - X_MIN) / (X_MAX - X_MIN)
    return int(PERIOD_MAX - x_norm * (PERIOD_MAX - PERIOD_MIN))


def period_to_x(period):
    """Map pulse period back to x value."""
    p_clipped = np.clip(period, PERIOD_MIN, PERIOD_MAX)
    x_norm = (PERIOD_MAX - p_clipped) / (PERIOD_MAX - PERIOD_MIN)
    return X_MIN + x_norm * (X_MAX - X_MIN)


def create_sub_file(x_values, filename=None):
    """Create Flipper .sub file from x trajectory."""
    pulses = []
    for x in x_values:
        p = x_to_period(x)
        pulses.extend([p // 2, -(p // 2)])

    content = (
        "Filetype: Flipper SubGhz RAW File\n"
        "Version: 1\n"
        "Frequency: 433920000\n"
        "Preset: FuriHalSubGhzPresetOok650Async\n"
        "Protocol: RAW\n"
        "RAW_Data: " + " ".join(map(str, pulses)) + "\n"
    )

    if filename:
        with open(filename, 'w') as f:
            f.write(content)
    return content


def text_to_bits(text):
    """Convert ASCII text to bit array."""
    bits = []
    for char in text:
        for i in range(8):
            bits.append((ord(char) >> (7 - i)) & 1)
    return np.array(bits)


def bits_to_text(bits):
    """Convert bit array to ASCII text."""
    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        if 32 <= byte < 127:
            chars.append(chr(byte))
    return ''.join(chars)


# =============================================================================
# HARDWARE INTERFACE
# =============================================================================

FLIPPER_PORT = '/dev/ttyACM0'
FLIPPER_BAUD = 115200
SDR_FREQ = 433920000
SDR_RATE = 1000000
SDR_GAIN = 40


def flipper_upload_and_tx(sub_content, capture_duration=5.0):
    """Upload .sub file to Flipper and transmit while capturing with SDR.

    CRITICAL: Must call `loader close` before tx_from_file or it fails with
    "this command cannot be run while an application is open"

    Returns path to capture file, or None on failure.
    """
    import sys
    capture_file = '/tmp/chaos_capture.bin'

    print("    [a] Writing temp file...", flush=True)
    with open('/tmp/chaos.sub', 'w') as f:
        f.write(sub_content)

    print("    [b] Opening serial...", flush=True)
    ser = serial.Serial(FLIPPER_PORT, FLIPPER_BAUD, timeout=2, write_timeout=2)
    time.sleep(0.3)
    ser.read(ser.in_waiting)

    print("    [c] loader close...", flush=True)
    ser.write(b'loader close\r\n')
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    print("    [d] storage remove...", flush=True)
    ser.write(b'storage remove /ext/subghz/chaos.sub\r\n')
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    print("    [e] storage write...", flush=True)
    ser.write(b'storage write /ext/subghz/chaos.sub\r\n')
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    print(f"    [f] Sending {len(sub_content)} bytes...", flush=True)
    with open('/tmp/chaos.sub', 'r') as f:
        content = f.read()
    for i in range(0, len(content), 64):
        ser.write(content[i:i+64].encode())
        time.sleep(0.02)

    print("    [g] Ctrl+C to end write...", flush=True)
    ser.write(b'\x03')
    time.sleep(0.3)
    ser.read(ser.in_waiting)

    print("    [h] loader close again...", flush=True)
    ser.write(b'loader close\r\n')
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    print("    [i] Starting SDR capture...", flush=True)
    n_samples = int(capture_duration * SDR_RATE)
    cap = subprocess.Popen(
        ['rtl_sdr', '-f', str(SDR_FREQ), '-s', str(SDR_RATE),
         '-g', str(SDR_GAIN), '-n', str(n_samples), capture_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(1.5)

    print("    [j] Transmitting...", flush=True)
    ser.write(b'subghz tx_from_file /ext/subghz/chaos.sub\r\n')
    time.sleep(3)
    response = ser.read(ser.in_waiting)
    ser.close()
    print("    [k] TX done", flush=True)

    if b'cannot be run' in response:
        print("ERROR: Flipper TX blocked - app still open")
        cap.terminate()
        return None

    print("    [l] Waiting for capture...", flush=True)
    try:
        cap.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("ERROR: Capture timed out")
        cap.kill()
        return None

    print("    [m] Done", flush=True)
    return capture_file


def extract_periods_from_capture(capture_file, expected_periods=None):
    """Extract pulse periods from SDR capture.

    Returns (periods_array, snr) or (None, snr) on failure.

    NOTE: Uses 450-sample smoothing window (not 100) to properly extract
    pulse envelope without detecting carrier oscillations.
    """
    raw = np.fromfile(capture_file, dtype=np.uint8)
    iq = (raw.astype(np.float32) - 127.5)
    mag = np.sqrt(iq[0::2]**2 + iq[1::2]**2)

    # Coarse envelope to find burst
    env = uniform_filter1d(mag, 1000)
    noise = np.percentile(env, 10)
    peak = np.percentile(env, 99)
    snr = peak / noise

    if snr < 5:
        return None, snr

    # Find burst start
    threshold = (noise + peak) / 3
    above = np.where(env > threshold)[0]

    if len(above) == 0:
        return None, snr

    burst_start = above[0]
    burst = mag[burst_start:burst_start + 2000000]

    # Heavy smoothing (450 samples) for pulse envelope
    # This is critical - smaller windows detect carrier oscillations
    heavy_env = uniform_filter1d(burst, 450)
    low = np.percentile(heavy_env, 15)
    high = np.percentile(heavy_env, 85)
    threshold = (low + high) / 2

    binary = (heavy_env > threshold).astype(int)
    rising = np.where(np.diff(binary) == 1)[0]
    periods = np.diff(rising)

    # Filter valid periods
    valid = (periods > PERIOD_MIN * 0.7) & (periods < PERIOD_MAX * 1.3)
    periods_clean = periods[valid]

    return periods_clean, snr


def find_best_alignment(periods_tx, periods_rx, max_offset=20):
    """Find best alignment between TX and RX periods using cross-correlation.

    Returns (offset, correlation) where offset is how many samples to skip
    in periods_rx to align with periods_tx.
    """
    n_tx = len(periods_tx)
    n_rx = len(periods_rx)

    best_corr = 0
    best_offset = 0

    for offset in range(-min(max_offset, n_rx-10), min(max_offset, n_rx-10)):
        if offset >= 0:
            tx_seg = periods_tx[:min(n_tx, n_rx-offset)]
            rx_seg = periods_rx[offset:offset+len(tx_seg)]
        else:
            tx_seg = periods_tx[-offset:-offset+min(n_tx+offset, n_rx)]
            rx_seg = periods_rx[:len(tx_seg)]

        if len(tx_seg) < 10:
            continue

        corr = np.corrcoef(tx_seg, rx_seg)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_offset = offset

    return best_offset, best_corr


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_sync_experiment(n_samples=80):
    """Run chaotic synchronization experiment over RF.

    Uses cross-correlation alignment and skips transient for accurate results.
    """
    print("=" * 60)
    print("CHAOTIC SYNCHRONIZATION OVER RF")
    print("=" * 60)

    # Generate trajectory with warmup to reach attractor
    np.random.seed(42)
    warmup = 1000
    total = warmup + n_samples
    t, traj = generate_trajectory(duration=total * DT, dt=DT)
    x_true = traj[warmup:warmup + n_samples, 0]
    y_true = traj[warmup:warmup + n_samples, 1]
    z_true = traj[warmup:warmup + n_samples, 2]

    print(f"\n[1] Generated {n_samples} samples, x range [{x_true.min():.2f}, {x_true.max():.2f}]")

    # Create and transmit
    sub_content = create_sub_file(x_true)
    print("[2] Uploading to Flipper...")
    print("[3] Starting SDR capture...")
    print("[4] Transmitting...")

    capture_file = flipper_upload_and_tx(sub_content)
    if capture_file is None:
        print("ERROR: TX failed")
        return None

    # Analyze
    print("[5] Analyzing...")
    periods_rx, snr = extract_periods_from_capture(capture_file, n_samples)
    print(f"    SNR: {snr:.1f}")

    if periods_rx is None or len(periods_rx) < n_samples // 2:
        print(f"    ERROR: Only {len(periods_rx) if periods_rx is not None else 0} periods recovered")
        return None

    print(f"    Periods recovered: {len(periods_rx)}")

    # Compute expected TX periods
    periods_tx = np.array([x_to_period(x) for x in x_true])

    # Find best alignment using cross-correlation
    offset, raw_corr = find_best_alignment(periods_tx, periods_rx)
    print(f"    Alignment: offset={offset}, raw_corr={raw_corr:.4f}")

    if raw_corr < 0.5:
        print("    ERROR: Poor alignment (raw_corr < 0.5)")
        return None

    # Apply alignment
    if offset >= 0:
        tx_aligned = periods_tx[:min(len(periods_tx), len(periods_rx)-offset)]
        rx_aligned = periods_rx[offset:offset+len(tx_aligned)]
    else:
        tx_aligned = periods_tx[-offset:-offset+min(len(periods_tx)+offset, len(periods_rx))]
        rx_aligned = periods_rx[:len(tx_aligned)]

    n = len(tx_aligned)

    # Calibrate: fit periods_rx = slope * periods_tx + intercept
    slope, intercept = np.polyfit(tx_aligned, rx_aligned, 1)
    print(f"    Calibration: slope={slope:.4f}, intercept={intercept:.1f}")

    # Apply calibration and convert to x
    periods_calibrated = (rx_aligned - intercept) / slope
    x_rx = np.array([period_to_x(p) for p in periods_calibrated])

    # Align true values
    x_true_aligned = x_true[:n] if offset >= 0 else x_true[-offset:-offset+n]
    y_true_aligned = y_true[:n] if offset >= 0 else y_true[-offset:-offset+n]
    z_true_aligned = z_true[:n] if offset >= 0 else z_true[-offset:-offset+n]

    # Run Pecora-Carroll sync with TRUE initial conditions
    y_sync, z_sync = pecora_carroll_sync(x_rx, DT, y_true_aligned[0], z_true_aligned[0])

    # Skip transient (first 15 samples) for correlation computation
    skip = 15
    x_corr = np.corrcoef(x_true_aligned[skip:], x_rx[skip:])[0, 1]
    y_corr = np.corrcoef(y_true_aligned[skip:], y_sync[skip:])[0, 1]
    z_corr = np.corrcoef(z_true_aligned[skip:], z_sync[skip:])[0, 1]

    print("\n" + "=" * 60)
    print("RESULTS (after transient skip)")
    print("=" * 60)
    print(f"x recovery correlation:  {x_corr:.4f}")
    print(f"y synchronization:       {y_corr:.4f}")
    print(f"z synchronization:       {z_corr:.4f}")

    if raw_corr > 0.5 and y_corr > 0.8:
        print("\n*** SYNC SUCCESSFUL ***")

    return {'x_corr': x_corr, 'y_corr': y_corr, 'z_corr': z_corr,
            'snr': snr, 'raw_corr': raw_corr, 'n_samples': n}


def run_modem_experiment(message="A"):
    """Run chaos-masked communication experiment over RF."""
    print("=" * 60)
    print("CHAOS MODEM OVER RF")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\"")
    print(f"Bits: {list(bits_tx)}")
    print(f"Samples: {n_samples}, Mask amplitude: {MASK_AMPLITUDE}")

    # Generate trajectory with warmup
    np.random.seed(42)
    warmup = 1000
    total = warmup + n_samples
    t, traj = generate_trajectory(duration=total * DT, dt=DT)
    x_true = traj[warmup:warmup + n_samples, 0]

    print(f"\n[1] Generated trajectory, x range [{x_true.min():.2f}, {x_true.max():.2f}]")

    # Add mask for bit=1
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    x_masked = x_true + mask

    # Create and transmit
    sub_content = create_sub_file(x_masked)
    print("[2] Uploading to Flipper...")
    print("[3] Starting SDR capture...")
    print("[4] Transmitting...")

    capture_file = flipper_upload_and_tx(sub_content)
    if capture_file is None:
        print("ERROR: TX failed")
        return None

    # Analyze
    print("[5] Analyzing...")
    periods, snr = extract_periods_from_capture(capture_file, n_samples)
    print(f"    SNR: {snr:.1f}")

    if periods is None:
        print("    ERROR: No signal detected")
        return None

    print(f"    Periods: {len(periods)} (expected {n_samples})")

    # Convert to x values (this is s = x + mask)
    n = min(len(periods), n_samples)
    s_recv = np.array([period_to_x(p) for p in periods[:n]])

    # Run Cuomo-Oppenheim observer to estimate x
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT)

    # Recover mask
    residual = s_recv - x_est

    # Decode bits using only settled portion (last half of each bit period)
    settle = SAMPLES_PER_BIT // 2
    bits_rx = []
    for i in range(len(bits_tx)):
        start = i * SAMPLES_PER_BIT + settle
        end = (i + 1) * SAMPLES_PER_BIT
        if end <= len(residual):
            seg = residual[start:end]
            bits_rx.append(1 if np.median(seg) > MASK_AMPLITUDE / 2 else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"TX: {list(bits_tx)}")
    print(f"RX: {list(bits_rx)}")
    print(f"Bit errors: {errors}/{len(bits_rx)}")

    # Per-bit analysis
    print("\nPer-bit residuals (settled portion):")
    for i in range(min(len(bits_tx), len(bits_rx))):
        start = i * SAMPLES_PER_BIT + settle
        end = min((i + 1) * SAMPLES_PER_BIT, len(residual))
        if end > start:
            seg = residual[start:end]
            expected = MASK_AMPLITUDE if bits_tx[i] == 1 else 0
            status = "OK" if bits_tx[i] == bits_rx[i] else "ERR"
            print(f"  bit {i}: tx={bits_tx[i]} rx={bits_rx[i]} "
                  f"residual={np.median(seg):+.2f} expected={expected:.0f} [{status}]")

    if errors == 0:
        decoded = bits_to_text(bits_rx)
        print(f"\n*** DECODED: \"{decoded}\" ***")

    return {'bits_tx': bits_tx, 'bits_rx': bits_rx, 'errors': errors, 'snr': snr}


def run_offline_test(message="A"):
    """Test modem encoding/decoding without RF (simulation only)."""
    print("=" * 60)
    print("OFFLINE MODEM TEST (no RF)")
    print("=" * 60)

    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    print(f"Message: \"{message}\" = {list(bits_tx)}")
    print(f"Samples: {n_samples}, Mask: {MASK_AMPLITUDE}")

    # Generate trajectory
    np.random.seed(42)
    warmup = 1000
    t, traj = generate_trajectory(duration=(warmup + n_samples) * DT, dt=DT)
    x_true = traj[warmup:warmup + n_samples, 0]

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    x_masked = x_true + mask

    # Simulate period encoding/decoding (ideal channel)
    periods = np.array([x_to_period(x) for x in x_masked])
    s_recv = np.array([period_to_x(p) for p in periods])

    # Run observer
    x_est, y_est, z_est = cuomo_oppenheim_observer(s_recv, DT)
    residual = s_recv - x_est

    # Decode using settled portion (last half of each bit)
    settle = SAMPLES_PER_BIT // 2
    bits_rx = []
    for i in range(len(bits_tx)):
        start = i * SAMPLES_PER_BIT + settle
        end = (i + 1) * SAMPLES_PER_BIT
        seg = residual[start:end]
        bits_rx.append(1 if np.median(seg) > MASK_AMPLITUDE / 2 else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx != bits_rx)

    # x reconstruction quality
    x_corr = np.corrcoef(x_true, x_est)[0, 1]

    print(f"\nx reconstruction correlation: {x_corr:.4f}")
    print(f"\nTX: {list(bits_tx)}")
    print(f"RX: {list(bits_rx)}")
    print(f"Errors: {errors}/{len(bits_tx)}")

    # Per-bit
    print("\nPer-bit analysis (settled portion):")
    for i in range(len(bits_tx)):
        start = i * SAMPLES_PER_BIT + settle
        end = (i + 1) * SAMPLES_PER_BIT
        seg = residual[start:end]
        expected = MASK_AMPLITUDE if bits_tx[i] == 1 else 0
        status = "OK" if bits_tx[i] == bits_rx[i] else "ERR"
        print(f"  bit {i}: tx={bits_tx[i]} rx={bits_rx[i]} "
              f"residual={np.median(seg):+.2f} expected={expected:.0f} [{status}]")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExamples:")
        print("  python chaos_rf.py sync 50      # 50-sample sync experiment")
        print("  python chaos_rf.py modem A      # Send letter 'A'")
        print("  python chaos_rf.py test         # Offline test")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "sync":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_sync_experiment(n)

    elif cmd == "modem":
        msg = sys.argv[2] if len(sys.argv) > 2 else "A"
        run_modem_experiment(msg)

    elif cmd == "test":
        msg = sys.argv[2] if len(sys.argv) > 2 else "A"
        run_offline_test(msg)

    else:
        print(f"Unknown command: {cmd}")
        print("Use: sync, modem, or test")
