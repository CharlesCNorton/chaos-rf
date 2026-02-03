# chaos-rf

Chaotic synchronization and communication over 433 MHz RF using a Flipper Zero transmitter and RTL-SDR receiver.

## What This Is

A working implementation of:

1. **Pecora-Carroll Chaotic Synchronization** — Transmit one variable of a Lorenz attractor over RF; the receiver reconstructs the other two variables without ever receiving them.

2. **Chaos-Masked Communication** — Hide binary data inside a chaotic carrier signal. The receiver extracts the message by synchronizing to the chaos and subtracting it out.

Both demonstrated over actual RF (433.92 MHz ISM band), not simulation.

## Results

### Chaotic Synchronization

| Metric | Value |
|--------|-------|
| x(t) recovery correlation | 0.9975 |
| y(t) sync correlation | 0.9546 |
| z(t) sync correlation | 0.9218 |
| Samples transmitted | 50 |
| RF frequency | 433.92 MHz |
| Modulation | Pulse period encodes x |

The transmitter sends only x(t). The receiver, knowing the Lorenz parameters, reconstructs y(t) and z(t) with >92% correlation.

### Chaos-Masked Data Transmission

| Metric | Value |
|--------|-------|
| Message | "A" (ASCII 0x41) |
| Bits transmitted | 8 |
| Bit errors | 0 |
| SNR | 61.2 |
| Mask amplitude | 3.0 |
| Samples per bit | 5 |

Binary data hidden in chaotic carrier, recovered perfectly.

## Theory

### Lorenz System

The Lorenz attractor is a system of three coupled differential equations:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

With standard parameters σ=10, ρ=28, β=8/3, the system exhibits deterministic chaos — sensitive dependence on initial conditions, but bounded to a strange attractor.

### Pecora-Carroll Synchronization (1990)

Key insight: If you transmit x(t) to a receiver running the same Lorenz system, and the receiver substitutes the received x for its local x in the y and z equations, the receiver's y and z will converge to the transmitter's y and z.

**Transmitter:**
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y      → transmit x(t)
dz/dt = xy - βz
```

**Receiver:**
```
x̂ = x_received            ← substitute received x
dŷ/dt = x̂(ρ - ẑ) - ŷ      → ŷ converges to y
dẑ/dt = x̂ŷ - βẑ           → ẑ converges to z
```

This works because the conditional Lyapunov exponents of the y-z subsystem driven by x are negative.

### Chaotic Masking

To transmit data:

1. Generate chaotic carrier x(t)
2. For bit=1: transmit x(t) + ε (small offset)
3. For bit=0: transmit x(t)
4. Receiver synchronizes to recover x̂(t) ≈ x(t)
5. Compute residual: r(t) = received - x̂(t)
6. Threshold residual to recover bits

An eavesdropper without the Lorenz parameters sees only noise-like chaos.

## Hardware

| Device | Role | Details |
|--------|------|---------|
| Flipper Zero | Transmitter | CC1101 radio, 433.92 MHz |
| RTL-SDR | Receiver | R820T tuner, 1 MS/s capture |
| Raspberry Pi 5 | Processing | Runs both devices, eco mode |

### Hardware Notes

- Pi must be in **eco mode** (1.5 GHz) to power both USB devices without over-current
- SDR sample rate: 1 MS/s (higher draws too much power)
- SDR gain: 30 (40 saturates, 20 loses signal)
- Devices should be within a few meters for reliable reception

## RF Encoding

The chaotic signal x(t) is encoded as pulse periods in a Flipper .sub file:

```
x value  →  period (μs)
─────────────────────────
x = -20  →  2000 μs (slow)
x = +20  →   500 μs (fast)
```

Linear mapping: `period = 2000 - ((x + 20) / 40) * 1500`

The .sub file format:
```
Filetype: Flipper SubGhz RAW File
Version: 1
Frequency: 433920000
Preset: FuriHalSubGhzPresetOok650Async
Protocol: RAW
RAW_Data: 605 -605 604 -604 602 -602 ...
```

Values are microseconds. Positive = carrier on, negative = carrier off.

## Code Structure

```
chaos-rf/
├── lorenz.py           # Lorenz system integrator
├── encoder.py          # x(t) → .sub file
├── capture.py          # SDR capture wrapper
├── demodulate.py       # Signal → periods → x(t)
├── receiver.py         # Pecora-Carroll synchronization
├── chaos_modem.py      # Message encode/decode
└── experiment.py       # Full TX/RX pipeline
```

### lorenz.py

```python
from scipy.integrate import odeint

SIGMA, RHO, BETA = 10.0, 28.0, 8/3

def lorenz(state, t):
    x, y, z = state
    return [
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z
    ]

def generate_trajectory(duration, dt, seed=42):
    np.random.seed(seed)
    initial = [1.0, 1.0, 1.0]
    t = np.arange(0, duration, dt)
    return t, odeint(lorenz, initial, t)
```

### receiver.py

```python
def pecora_carroll_sync(x_received, dt, y0=1.0, z0=1.0):
    """Reconstruct y, z from received x."""
    n = len(x_received)
    y, z = np.zeros(n), np.zeros(n)
    y[0], z[0] = y0, z0

    for i in range(1, n):
        x = x_received[i-1]
        dy = x * (RHO - z[i-1]) - y[i-1]
        dz = x * y[i-1] - BETA * z[i-1]
        y[i] = y[i-1] + dy * dt
        z[i] = z[i-1] + dz * dt

    return y, z
```

## Running the Experiments

### Prerequisites

```bash
# On Raspberry Pi
sudo apt install rtl-sdr python3-numpy python3-scipy

# Verify SDR
rtl_test -t

# Verify Flipper
ls /dev/ttyACM0
```

### Chaotic Synchronization

```bash
cd chaos-rf

# Generate .sub file with chaotic signal
python3 encoder.py

# Upload to Flipper (via serial or qFlipper)
# Then run:
python3 experiment.py
```

### Chaos Modem

```bash
# Encode message into chaos
python3 chaos_modem.py encode "HELLO" > message.sub

# Upload message.sub to Flipper
# Capture with SDR while transmitting
# Decode:
python3 chaos_modem.py decode capture.bin
```

## SDR Signal Processing

The received signal requires envelope detection to extract timing:

```python
from scipy.ndimage import uniform_filter1d

# Load IQ capture
raw = np.fromfile('capture.bin', dtype=np.uint8)
iq = (raw.astype(np.float32) - 127.5)
mag = np.sqrt(iq[0::2]**2 + iq[1::2]**2)

# Coarse envelope (find bursts)
envelope = uniform_filter1d(mag, size=5000)

# Fine envelope (find pulses within burst)
burst_smooth = uniform_filter1d(burst, size=100)

# Extract pulse timing
rising_edges = np.where(np.diff((burst_smooth > threshold).astype(int)) == 1)[0]
periods = np.diff(rising_edges)  # In microseconds at 1 MS/s
```

Two-stage smoothing: coarse (5ms window) finds where the signal is, fine (100μs window) extracts individual pulse timing.

## Flipper Serial Gotchas

The Flipper's serial CLI has quirks:

| What | Status |
|------|--------|
| Single `subghz tx` | Works |
| Rapid sequential TX | Blocks unpredictably |
| `storage write` large files | Often hangs |
| `tx_from_file` | Reliable for playback |
| USB reset before use | Recommended: `sudo usbreset 0483:5740` |

Best practice: Pre-generate .sub files, upload once, use `tx_from_file`.

## References

- L. M. Pecora and T. L. Carroll, "Synchronization in Chaotic Systems," Physical Review Letters, vol. 64, no. 8, pp. 821-824, 1990.

- K. M. Cuomo and A. V. Oppenheim, "Circuit Implementation of Synchronized Chaos with Applications to Communications," Physical Review Letters, vol. 71, no. 1, pp. 65-68, 1993.

- L. Kocarev and U. Parlitz, "General Approach for Chaotic Synchronization with Applications to Communication," Physical Review Letters, vol. 74, no. 25, pp. 5028-5031, 1995.

## License

MIT
