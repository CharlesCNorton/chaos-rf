"""Demodulate captured RF signal to recover x(t) trajectory."""
import numpy as np
from capture import load_iq, SAMPLE_RATE

# Must match encoder parameters
PERIOD_MIN_US = 100
PERIOD_MAX_US = 400
X_MIN = -20.0
X_MAX = 20.0

def envelope_detect(iq, window=50):
    """Compute signal envelope (magnitude)."""
    return np.abs(iq)

def find_pulse_edges(envelope, threshold=None):
    """Find rising and falling edges in envelope.
    
    Returns arrays of sample indices for rising and falling edges.
    """
    if threshold is None:
        # Adaptive threshold: midpoint between noise floor and peak
        noise_floor = np.percentile(envelope, 10)
        peak = np.percentile(envelope, 90)
        threshold = (noise_floor + peak) / 2
    
    # Binary signal
    binary = (envelope > threshold).astype(int)
    
    # Find edges
    diff = np.diff(binary)
    rising = np.where(diff == 1)[0]
    falling = np.where(diff == -1)[0]
    
    return rising, falling, threshold

def extract_periods(rising_edges, sample_rate=SAMPLE_RATE):
    """Extract pulse periods from rising edge times.
    
    Each complete cycle is rising edge to rising edge.
    Returns periods in microseconds.
    """
    if len(rising_edges) < 2:
        return np.array([])
    
    # Period = time between consecutive rising edges
    periods_samples = np.diff(rising_edges)
    periods_us = periods_samples * 1e6 / sample_rate
    
    return periods_us

def period_to_x(period_us):
    """Convert pulse period to x value (inverse of encoder mapping)."""
    # Linear mapping: period_max -> x_min, period_min -> x_max
    period_clipped = np.clip(period_us, PERIOD_MIN_US, PERIOD_MAX_US)
    normalized = (PERIOD_MAX_US - period_clipped) / (PERIOD_MAX_US - PERIOD_MIN_US)
    x = X_MIN + normalized * (X_MAX - X_MIN)
    return x

def demodulate(iq_file):
    """Full demodulation pipeline.
    
    Returns:
        x_recovered: Recovered x(t) trajectory
        info: Dictionary with diagnostic info
    """
    # Load IQ data
    iq = load_iq(iq_file)
    
    # Envelope detection
    env = envelope_detect(iq)
    
    # Find pulse edges
    rising, falling, threshold = find_pulse_edges(env)
    
    # Extract periods
    periods = extract_periods(rising)
    
    # Filter outliers (noise spikes, missed edges)
    valid = (periods > PERIOD_MIN_US * 0.5) & (periods < PERIOD_MAX_US * 2)
    periods_clean = periods[valid]
    
    # Convert to x values
    x_recovered = period_to_x(periods_clean)
    
    info = {
        "samples": len(iq),
        "threshold": threshold,
        "rising_edges": len(rising),
        "falling_edges": len(falling),
        "periods_raw": len(periods),
        "periods_valid": len(periods_clean),
        "period_range": (periods_clean.min(), periods_clean.max()) if len(periods_clean) > 0 else (0, 0),
        "x_range": (x_recovered.min(), x_recovered.max()) if len(x_recovered) > 0 else (0, 0)
    }
    
    return x_recovered, info

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "capture.bin"
    
    x, info = demodulate(filename)
    print(f"Demodulation results:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\nRecovered {len(x)} x values")
    if len(x) > 0:
        print(f"First 10: {x[:10]}")
