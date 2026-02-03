# chaos-rf

Chaotic synchronization and communication over 433 MHz RF using Flipper Zero + RTL-SDR.

## What This Does

Transmits one variable (x) of a Lorenz attractor over RF. The receiver reconstructs the other two variables (y, z) using Pecora-Carroll synchronization. Binary data can be hidden in the chaotic carrier using Cuomo-Oppenheim masking.

## Results

### Synchronization
| Metric | Value |
|--------|-------|
| x recovery | 0.956 |
| y sync | 0.997 |
| z sync | 0.959 |

### Modem (150 samples/bit)
| Metric | Value |
|--------|-------|
| Bit errors | 0/8 |
| SNR | 30 |

## Hardware

- Raspberry Pi 5 (eco mode required for dual USB)
- Flipper Zero (CC1101, 433.92 MHz)
- RTL-SDR (R820T, gain=40)

Both devices on the Pi via USB within a few meters of each other.

## Usage

```bash
python3 chaos_rf.py sync 50      # Sync experiment
python3 chaos_rf.py modem A      # Send 'A' over RF
python3 chaos_rf.py test         # Offline test
```

## Theory

**Lorenz system:**
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

**Pecora-Carroll sync:** Transmit x(t). Receiver integrates y,z driven by received x. After transient, receiver's y,z converge to transmitter's y,z.

**Cuomo-Oppenheim masking:** Add offset m to x for bit=1. Receiver runs observer where x evolves via natural dynamics (filtering the mask) while y,z are driven by s=x+m. Recover m = s - x_est.

**Key parameter:** 150 samples/bit (0.75 Lorenz time = 7.5 time constants) allows observer to settle between bits.

## Flipper Serial Notes

Large uploads (>10KB) require:
- 16-byte chunks
- 100ms delays between chunks
- Periodic buffer drain
- `loader close` before `tx_from_file`
- USB reset between experiments

## References

- Pecora & Carroll, PRL 1990
- Cuomo & Oppenheim, PRL 1993

## License

MIT
