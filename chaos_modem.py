"""Chaos-masked communication modem.

Encodes binary data into a chaotic carrier signal using additive masking.
The receiver extracts the message by synchronizing to the chaos.
"""
import numpy as np
from lorenz import generate_trajectory
from receiver import pecora_carroll_sync

# Modulation parameters
SAMPLES_PER_BIT = 5      # Chaotic samples per data bit
MASK_AMPLITUDE = 3.0     # Offset added for bit=1
PERIOD_MIN = 500         # Minimum pulse period (us)
PERIOD_MAX = 2000        # Maximum pulse period (us)
X_MIN, X_MAX = -20.0, 20.0


def text_to_bits(text):
    """Convert ASCII text to bit array."""
    bits = []
    for char in text:
        for i in range(8):
            bits.append((ord(char) >> (7 - i)) & 1)
    return np.array(bits)


def bits_to_text(bits):
    """Convert bit array to ASCII text."""
    bits = np.array(bits)
    pad = (8 - len(bits) % 8) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad)])

    chars = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        if 32 <= byte < 127:
            chars.append(chr(byte))
    return ''.join(chars)


def x_to_period(x):
    """Map Lorenz x value to pulse period in microseconds."""
    x_norm = (np.clip(x, X_MIN, X_MAX) - X_MIN) / (X_MAX - X_MIN)
    return int(PERIOD_MAX - x_norm * (PERIOD_MAX - PERIOD_MIN))


def period_to_x(p):
    """Map pulse period back to x value."""
    p_clipped = np.clip(p, PERIOD_MIN, PERIOD_MAX)
    x_norm = (PERIOD_MAX - p_clipped) / (PERIOD_MAX - PERIOD_MIN)
    return X_MIN + x_norm * (X_MAX - X_MIN)


def encode_message(message, seed=42, dt=0.02):
    """Encode message into chaos-masked signal.

    Args:
        message: ASCII string to encode
        seed: Random seed for Lorenz trajectory (must match decoder)
        dt: Integration timestep

    Returns:
        x_masked: Chaotic signal with message embedded
        x_true: Original chaotic signal (for reference)
        bits: Message as bit array
    """
    bits = text_to_bits(message)
    n_samples = len(bits) * SAMPLES_PER_BIT

    # Generate chaotic carrier
    np.random.seed(seed)
    t, traj = generate_trajectory(duration=n_samples * dt, dt=dt)
    x_true = traj[:n_samples, 0]
    y_true = traj[:n_samples, 1]
    z_true = traj[:n_samples, 2]

    # Create masking signal
    mask = np.zeros(n_samples)
    for i, bit in enumerate(bits):
        start = i * SAMPLES_PER_BIT
        end = start + SAMPLES_PER_BIT
        if bit == 1:
            mask[start:end] = MASK_AMPLITUDE

    x_masked = x_true + mask

    return x_masked, x_true, y_true, z_true, bits


def generate_sub_file(x_values, filename=None):
    """Generate Flipper .sub file from x values.

    Args:
        x_values: Array of Lorenz x values (possibly masked)
        filename: Output filename (if None, returns string)

    Returns:
        .sub file content as string
    """
    pulses = []
    for x in x_values:
        period = x_to_period(x)
        half = period // 2
        pulses.extend([half, -half])

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


def decode_from_periods(periods, seed=42, dt=0.02):
    """Decode message from recovered pulse periods.

    Args:
        periods: Array of pulse periods in microseconds
        seed: Random seed (must match encoder)
        dt: Integration timestep (must match encoder)

    Returns:
        bits: Recovered bit array
        residual: Difference between received and sync'd chaos
    """
    # Recover x values from periods
    x_received = np.array([period_to_x(p) for p in periods])
    n_samples = len(x_received)

    # Regenerate the same chaotic trajectory
    np.random.seed(seed)
    t, traj = generate_trajectory(duration=n_samples * dt, dt=dt)
    x_sync = traj[:n_samples, 0]

    # Compute residual (should reveal the mask)
    residual = x_received - x_sync

    # Decode bits from residual
    n_bits = n_samples // SAMPLES_PER_BIT
    bits = []
    for i in range(n_bits):
        start = i * SAMPLES_PER_BIT
        end = start + SAMPLES_PER_BIT
        segment_mean = np.mean(residual[start:end])
        bits.append(1 if segment_mean > MASK_AMPLITUDE / 2 else 0)

    return np.array(bits), residual


def decode_with_sync(x_received, x_synced):
    """Decode using synchronized chaos for subtraction.

    Args:
        x_received: Received (masked) x values
        x_synced: Synchronized chaos (unmasked)

    Returns:
        bits: Recovered bit array
    """
    residual = x_received - x_synced
    n_bits = len(x_received) // SAMPLES_PER_BIT

    bits = []
    for i in range(n_bits):
        start = i * SAMPLES_PER_BIT
        end = start + SAMPLES_PER_BIT
        segment_mean = np.mean(residual[start:end])
        bits.append(1 if segment_mean > MASK_AMPLITUDE / 2 else 0)

    return np.array(bits)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python chaos_modem.py encode <message> [output.sub]")
        print("  python chaos_modem.py test <message>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "encode":
        message = sys.argv[2] if len(sys.argv) > 2 else "HELLO"
        outfile = sys.argv[3] if len(sys.argv) > 3 else None

        x_masked, x_true, y_true, z_true, bits = encode_message(message)
        sub_content = generate_sub_file(x_masked, outfile)

        if outfile:
            print(f"Encoded '{message}' to {outfile}")
            print(f"  Bits: {len(bits)}")
            print(f"  Samples: {len(x_masked)}")
        else:
            print(sub_content)

    elif cmd == "test":
        message = sys.argv[2] if len(sys.argv) > 2 else "HELLO"

        print(f"Testing chaos modem with message: '{message}'")

        # Encode
        x_masked, x_true, y_true, z_true, bits_tx = encode_message(message)
        print(f"  Encoded {len(bits_tx)} bits into {len(x_masked)} samples")

        # Simulate: convert to periods and back (ideal channel)
        periods = np.array([x_to_period(x) for x in x_masked])

        # Decode
        bits_rx, residual = decode_from_periods(periods)
        decoded = bits_to_text(bits_rx)

        errors = np.sum(bits_tx != bits_rx[:len(bits_tx)])
        print(f"  Bit errors: {errors}/{len(bits_tx)}")
        print(f"  Decoded: '{decoded}'")

        if errors == 0:
            print("  SUCCESS")
