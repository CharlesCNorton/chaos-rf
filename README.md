# chaos-rf

Chaotic synchronization and communication over 433 MHz RF using Flipper Zero + RTL-SDR.

## What This Does

Transmits one variable (x) of a Lorenz attractor over RF. The receiver reconstructs the other two variables (y, z) using Pecora-Carroll synchronization. Binary data can be hidden in the chaotic carrier using Cuomo-Oppenheim masking.

## Statistical Results

### Synchronization (N=10 trials)
| Metric | Value |
|--------|-------|
| Success rate | 70% |
| Failure mode | Poor alignment (raw_corr < 0.5) |

| Metric | Mean | Range |
|--------|------|-------|
| x recovery | 0.935 | [0.909, 0.955] |
| y sync | 0.997 | [0.997, 0.998] |
| z sync | 0.999 | [0.999, 0.999] |

### Chaos Modem (N=10 trials)
| Metric | Value |
|--------|-------|
| Success rate | 100% |
| Message | "A" (8 bits) |
| Bit errors | 0/8 all trials |

## Hardware

- Raspberry Pi 5 (eco mode required for dual USB power)
- Flipper Zero (CC1101, 433.92 MHz)
- RTL-SDR (R820T, gain=30, 1 MHz sample rate)

Both devices on the Pi via USB within a few meters of each other.

## Usage

```bash
python3 chaos_rf.py sync 80      # Sync experiment (80 samples)
python3 chaos_rf.py modem A      # Send 'A' over RF
python3 chaos_rf.py test         # Offline test
```

## Theory

**Lorenz system:**
```
dx/dt = sigma*(y - x)
dy/dt = x*(rho - z) - y
dz/dt = x*y - beta*z
```

**Pecora-Carroll sync:** Transmit x(t). Receiver integrates y,z driven by received x. After transient (~15 samples), receiver's y,z converge to transmitter's y,z.

**Cuomo-Oppenheim masking:** Add offset m to x for bit=1. Receiver runs observer where x evolves via natural dynamics (filtering the mask) while y,z are driven by s=x+m. Recover m = s - x_est.

## Critical Implementation Details

### Period Extraction
- Use 450-sample smoothing window (not 100 or 200)
- Smaller windows detect carrier oscillations, not pulse envelope
- This is the most common source of errors

### Alignment & Calibration
- TX and RX period sequences may be offset by several samples
- Use cross-correlation to find best alignment before calibration
- Linear calibration (slope, intercept) compensates for timing drift
- Reject trials with raw_corr < 0.5 (sync) or < 0.3 (modem)

### Transient
- Skip first 15 samples when computing sync correlations
- The receiver y,z subsystem needs time to converge to attractor

### Initial Conditions
- Use true y[0], z[0] for sync and modem (not arbitrary values)
- With random ICs, y sync drops from 0.99 to ~0.80

### Modem Decoding
- Use 80% settle time per bit (observer transient after mask)
- Threshold at 45% of mask amplitude
- Post-process: correct 1→0 transients (consecutive 1s with mid-range residual)
- 50 samples/bit at 4KB serial limit

## Flipper Serial Notes

Large uploads (>10KB) require:
- 64-byte chunks
- 20ms delays between chunks
- `loader close` before `tx_from_file`

## References

- Pecora & Carroll, "Synchronization in Chaotic Systems", PRL 1990
- Cuomo & Oppenheim, "Circuit Implementation of Synchronized Chaos", PRL 1993

## License

MIT
