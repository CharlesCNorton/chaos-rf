# chaos-rf

Chaotic synchronization over 433 MHz RF using a Flipper Zero transmitter and RTL-SDR receiver.

## What This Demonstrates

**Pecora-Carroll Chaotic Synchronization (1990)**: Transmit one variable (x) of a Lorenz attractor over RF. The receiver, knowing only the Lorenz parameters, reconstructs the other two variables (y, z) without ever receiving them.

This is a genuine physics demonstration - the receiver's y and z converge to the transmitter's y and z through the mathematical properties of chaotic systems.

## Verified Results

### Synchronization Experiment

| Metric | Value |
|--------|-------|
| x recovery correlation | 0.956 |
| y synchronization | 0.997 |
| z synchronization | 0.959 |
| SNR | 25+ |

The receiver successfully reconstructs y(t) and z(t) from transmitted x(t) alone.

### Chaos-Masked Modem

| Metric | Value |
|--------|-------|
| Bit error rate | ~25-30% |
| Period recovery | 99%+ |

The modem demonstrates the principle but has practical limitations due to observer dynamics (see Theory section).

## Hardware

| Device | Role | Notes |
|--------|------|-------|
| Raspberry Pi 5 | Host | Must be in **eco mode** (1.5 GHz) for USB power budget |
| Flipper Zero | Transmitter | CC1101 radio, 433.92 MHz |
| RTL-SDR | Receiver | R820T tuner, gain=40 |

Both USB devices connected to the Pi. They must be within a few meters of each other.

## Quick Start

```bash
# On Raspberry Pi
cd chaos-rf
pip install numpy scipy pyserial

# Run synchronization experiment (50 samples)
python3 chaos_rf.py sync 50

# Run chaos modem (send letter 'A')
python3 chaos_rf.py modem A

# Offline test (no RF hardware needed)
python3 chaos_rf.py test
```

## Theory

### Lorenz System

The Lorenz attractor is defined by:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

With standard parameters σ=10, ρ=28, β=8/3, the system exhibits deterministic chaos.

### Pecora-Carroll Synchronization

Key insight: If you transmit x(t) to a receiver running the y-z subsystem driven by the received x, the receiver's y and z will converge to the transmitter's y and z.

**Transmitter:**
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y     → transmit x(t)
dz/dt = xy - βz
```

**Receiver:**
```
x̂ = x_received
dŷ/dt = x̂(ρ - ẑ) - ŷ    → ŷ converges to y
dẑ/dt = x̂ŷ - βẑ         → ẑ converges to z
```

This works because the conditional Lyapunov exponents of the y-z subsystem driven by x are negative.

### Chaos Masking (Cuomo-Oppenheim 1993)

To transmit data hidden in chaos:

1. Generate chaotic carrier x(t)
2. For bit=1: transmit s(t) = x(t) + m (add offset)
3. For bit=0: transmit s(t) = x(t)
4. Receiver estimates x̂(t) using an observer
5. Recover mask: m = s(t) - x̂(t)

**Observer equations:**
```
dx̂/dt = σ(ŷ - x̂)        ← natural dynamics (filters mask)
dŷ/dt = s(ρ - ẑ) - ŷ    ← driven by received s
dẑ/dt = sŷ - βẑ         ← driven by received s
```

The x equation acts as a low-pass filter: x̂ tracks x, not s=x+m.

**Practical limitation:** The mask perturbs the trajectory. After a bit transition, the observer takes time to settle, causing errors on subsequent bits. This is a known limitation of discrete digital implementations vs the original analog circuits.

## RF Encoding

The chaotic signal x(t) is encoded as pulse periods:

```
x value  →  period (μs)
─────────────────────────
x = -20  →  2000 μs (slow)
x = +20  →   500 μs (fast)
```

The .sub file format for Flipper:
```
Filetype: Flipper SubGhz RAW File
Version: 1
Frequency: 433920000
Preset: FuriHalSubGhzPresetOok650Async
Protocol: RAW
RAW_Data: 605 -605 604 -604 ...
```

Values are microseconds. Positive = carrier on, negative = carrier off.

## Critical Implementation Notes

### Flipper Serial Interface

**CRITICAL:** Before calling `subghz tx_from_file`, you MUST call `loader close` or TX will fail with "this command cannot be run while an application is open".

```python
ser.write(b'loader close\r\n')   # Close any open apps
time.sleep(0.2)
ser.write(b'storage write /ext/subghz/chaos.sub\r\n')
# ... upload file content ...
ser.write(b'\x03')  # Ctrl+C to end write
time.sleep(0.3)
ser.write(b'loader close\r\n')   # Close storage app
time.sleep(0.2)
ser.write(b'subghz tx_from_file /ext/subghz/chaos.sub\r\n')
```

The `storage write` command opens an app that must be closed before TX.

### SDR Settings

- **Gain = 40** (not 30) - higher gain needed for adequate SNR
- **Sample rate = 1 MS/s** - higher rates may exceed USB power budget in eco mode
- **Start capture BEFORE transmitting** - allow 1.5s for SDR to stabilize

### Pi Power Mode

With both Flipper and SDR connected via USB, the Pi 5 **must** be in eco mode (1.5 GHz) or USB over-current protection will trip. Check with:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# Should show 1500000
```

### Signal Processing

Two-stage envelope detection:
1. **Coarse** (5ms window): Find where the burst is
2. **Fine** (100μs window): Extract individual pulse edges

```python
env = uniform_filter1d(mag, 5000)      # Coarse
fine_env = uniform_filter1d(burst, 100) # Fine
```

## File Structure

```
chaos-rf/
├── chaos_rf.py     # Single script with everything
└── README.md       # This file
```

## References

- L. M. Pecora and T. L. Carroll, "Synchronization in Chaotic Systems," Physical Review Letters, vol. 64, no. 8, pp. 821-824, 1990.

- K. M. Cuomo and A. V. Oppenheim, "Circuit Implementation of Synchronized Chaos with Applications to Communications," Physical Review Letters, vol. 71, no. 1, pp. 65-68, 1993.

## License

MIT
