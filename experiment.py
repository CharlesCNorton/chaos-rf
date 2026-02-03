"""Full chaos synchronization and communication experiment.

Runs the complete TX/RX pipeline:
1. Generate Lorenz trajectory (optionally with message mask)
2. Encode as .sub file
3. Upload to Flipper and transmit
4. Capture with RTL-SDR
5. Demodulate and decode
"""
import serial
import subprocess
import time
import numpy as np
from scipy.ndimage import uniform_filter1d
from lorenz import generate_trajectory
from receiver import pecora_carroll_sync, synchronization_error
from chaos_modem import (
    text_to_bits, bits_to_text, x_to_period, period_to_x,
    SAMPLES_PER_BIT, MASK_AMPLITUDE, PERIOD_MIN, PERIOD_MAX
)

# Hardware configuration
FLIPPER_PORT = '/dev/ttyACM0'
FLIPPER_BAUD = 115200
SDR_FREQ = 433920000
SDR_RATE = 1000000
SDR_GAIN = 30


def upload_sub_file(sub_content, remote_path='/ext/subghz/chaos.sub'):
    """Upload .sub file to Flipper via serial."""
    ser = serial.Serial(FLIPPER_PORT, FLIPPER_BAUD, timeout=2)
    time.sleep(0.3)
    ser.read(ser.in_waiting)

    # Delete old file
    ser.write(f"storage remove {remote_path}\r\n".encode())
    time.sleep(0.2)
    ser.read(ser.in_waiting)

    # Write new file
    ser.write(f"storage write {remote_path}\r\n".encode())
    time.sleep(0.1)

    # Send in small chunks
    for i in range(0, len(sub_content), 32):
        ser.write(sub_content[i:i+32].encode())
        time.sleep(0.01)

    ser.write(b'\x03')
    time.sleep(0.3)
    ser.read(ser.in_waiting)
    ser.close()


def transmit_file(remote_path='/ext/subghz/chaos.sub'):
    """Trigger Flipper to transmit a .sub file."""
    ser = serial.Serial(FLIPPER_PORT, FLIPPER_BAUD, timeout=2)
    time.sleep(0.3)
    ser.read(ser.in_waiting)

    ser.write(f"subghz tx_from_file {remote_path}\r\n".encode())
    time.sleep(0.5)
    ser.read(ser.in_waiting)
    ser.close()


def capture_signal(duration_sec, output_file='capture.bin'):
    """Capture RF signal with RTL-SDR."""
    n_samples = int(duration_sec * SDR_RATE)
    cmd = [
        'rtl_sdr',
        '-f', str(SDR_FREQ),
        '-s', str(SDR_RATE),
        '-g', str(SDR_GAIN),
        '-n', str(n_samples),
        output_file
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def extract_periods(capture_file):
    """Extract pulse periods from SDR capture."""
    raw = np.fromfile(capture_file, dtype=np.uint8)
    iq = (raw.astype(np.float32) - 127.5)
    mag = np.sqrt(iq[0::2]**2 + iq[1::2]**2)

    # Coarse envelope to find burst
    envelope = uniform_filter1d(mag, size=5000)
    noise = np.percentile(envelope, 10)
    peak = np.percentile(envelope, 99)
    threshold = (noise + peak) / 2

    # Find burst start
    binary = (envelope > threshold).astype(int)
    rising = np.where(np.diff(binary) == 1)[0]

    if len(rising) == 0:
        return np.array([]), {'snr': peak / noise, 'bursts': 0}

    # Extract burst region
    start = max(0, rising[0] - 5000)
    burst = mag[start:start + 150000]

    # Fine envelope to find pulses
    burst_smooth = uniform_filter1d(burst, size=100)
    thresh2 = (np.percentile(burst_smooth, 20) + np.percentile(burst_smooth, 95)) / 2

    pulse_rising = np.where(np.diff((burst_smooth > thresh2).astype(int)) == 1)[0]
    periods = np.diff(pulse_rising)

    # Filter valid periods
    valid = (periods > PERIOD_MIN * 0.5) & (periods < PERIOD_MAX * 1.5)
    periods_clean = periods[valid]

    return periods_clean, {
        'snr': peak / noise,
        'bursts': len(rising),
        'pulses': len(pulse_rising),
        'valid_periods': len(periods_clean)
    }


def run_sync_experiment(n_samples=50, seed=42, dt=0.02):
    """Run chaotic synchronization experiment."""
    print("=" * 60)
    print("CHAOTIC SYNCHRONIZATION EXPERIMENT")
    print("=" * 60)

    # Generate trajectory
    print("\n[1] Generating Lorenz trajectory...")
    np.random.seed(seed)
    t, traj = generate_trajectory(duration=n_samples * dt, dt=dt)
    x_true = traj[:n_samples, 0]
    y_true = traj[:n_samples, 1]
    z_true = traj[:n_samples, 2]

    # Create .sub file
    pulses = []
    for x in x_true:
        p = x_to_period(x)
        pulses.extend([p // 2, -(p // 2)])

    sub_content = (
        "Filetype: Flipper SubGhz RAW File\n"
        "Version: 1\n"
        "Frequency: 433920000\n"
        "Preset: FuriHalSubGhzPresetOok650Async\n"
        "Protocol: RAW\n"
        "RAW_Data: " + " ".join(map(str, pulses)) + "\n"
    )

    # Start capture
    print("\n[2] Starting SDR capture...")
    cap = capture_signal(3.0, 'sync_capture.bin')
    time.sleep(1.5)

    # Upload and transmit
    print("\n[3] Uploading to Flipper...")
    upload_sub_file(sub_content)

    print("\n[4] Transmitting...")
    transmit_file()

    # Wait for capture
    print("\n[5] Waiting for capture...")
    cap.wait(timeout=10)

    # Analyze
    print("\n[6] Analyzing...")
    periods, info = extract_periods('sync_capture.bin')
    print(f"    SNR: {info['snr']:.1f}")
    print(f"    Periods recovered: {len(periods)}")

    if len(periods) < n_samples - 5:
        print("    ERROR: Too few periods recovered")
        return None

    # Recover x and run sync
    x_recv = np.array([period_to_x(p) for p in periods[:n_samples]])

    np.random.seed(seed)
    t2, traj2 = generate_trajectory(duration=n_samples * dt, dt=dt)
    x_sync = traj2[:n_samples, 0]

    x_corr = np.corrcoef(x_recv, x_sync)[0, 1]
    print(f"    x correlation: {x_corr:.4f}")

    # Run Pecora-Carroll
    y_recv, z_recv = pecora_carroll_sync(x_recv, dt, y0=5.0, z0=25.0)
    err = synchronization_error(y_true, z_true, y_recv, z_recv, skip_transient=10)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"x correlation:    {x_corr:.4f}")
    print(f"y sync:           {err['correlation_y']:.4f}")
    print(f"z sync:           {err['correlation_z']:.4f}")

    return {'x_corr': x_corr, 'y_corr': err['correlation_y'], 'z_corr': err['correlation_z']}


def run_modem_experiment(message="A", seed=42, dt=0.02):
    """Run chaos-masked communication experiment."""
    print("=" * 60)
    print("CHAOS MODEM EXPERIMENT")
    print("=" * 60)
    print(f"Message: '{message}'")

    # Encode message
    bits_tx = text_to_bits(message)
    n_samples = len(bits_tx) * SAMPLES_PER_BIT

    np.random.seed(seed)
    t, traj = generate_trajectory(duration=n_samples * dt, dt=dt)
    x_true = traj[:n_samples, 0]

    # Add mask
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits_tx):
        if bit == 1:
            mask[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT] = MASK_AMPLITUDE

    x_masked = x_true + mask

    # Create .sub file
    pulses = []
    for x in x_masked:
        p = x_to_period(x)
        pulses.extend([p // 2, -(p // 2)])

    sub_content = (
        "Filetype: Flipper SubGhz RAW File\n"
        "Version: 1\n"
        "Frequency: 433920000\n"
        "Preset: FuriHalSubGhzPresetOok650Async\n"
        "Protocol: RAW\n"
        "RAW_Data: " + " ".join(map(str, pulses)) + "\n"
    )

    # TX/RX
    print("\n[1] Starting capture...")
    cap = capture_signal(3.0, 'modem_capture.bin')
    time.sleep(1.5)

    print("[2] Uploading...")
    upload_sub_file(sub_content)

    print("[3] Transmitting...")
    transmit_file()

    cap.wait(timeout=10)

    # Analyze
    print("[4] Analyzing...")
    periods, info = extract_periods('modem_capture.bin')
    print(f"    SNR: {info['snr']:.1f}")
    print(f"    Periods: {len(periods)}")

    if len(periods) < n_samples:
        print("    WARNING: Fewer periods than expected")

    # Decode
    n = min(len(periods), n_samples)
    x_recv = np.array([period_to_x(p) for p in periods[:n]])

    np.random.seed(seed)
    t2, traj2 = generate_trajectory(duration=n * dt, dt=dt)
    x_sync = traj2[:n, 0]

    residual = x_recv - x_sync

    n_bits = n // SAMPLES_PER_BIT
    bits_rx = []
    for i in range(n_bits):
        seg = np.mean(residual[i * SAMPLES_PER_BIT:(i + 1) * SAMPLES_PER_BIT])
        bits_rx.append(1 if seg > MASK_AMPLITUDE / 2 else 0)

    bits_rx = np.array(bits_rx)
    errors = np.sum(bits_tx[:len(bits_rx)] != bits_rx)
    decoded = bits_to_text(bits_rx)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"TX: '{message}'  bits: {list(bits_tx)}")
    print(f"RX: '{decoded}'  bits: {list(bits_rx)}")
    print(f"Bit errors: {errors}/{len(bits_rx)}")

    if errors == 0:
        print("\n*** SUCCESS ***")

    return {'message': message, 'decoded': decoded, 'errors': errors}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python experiment.py sync [n_samples]")
        print("  python experiment.py modem [message]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "sync":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_sync_experiment(n_samples=n)

    elif cmd == "modem":
        msg = sys.argv[2] if len(sys.argv) > 2 else "A"
        run_modem_experiment(message=msg)
