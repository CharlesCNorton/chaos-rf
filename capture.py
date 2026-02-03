"""Capture RF signal from SDR for chaotic sync experiment."""
import subprocess
import numpy as np
import os

FREQ_HZ = 433920000  # 433.92 MHz
SAMPLE_RATE = 1000000  # 1 MS/s (reduced for USB power budget)
GAIN = 30  # RF gain (not too high to avoid saturation)

def capture(duration_sec, output_file="capture.bin"):
    """Capture IQ data from RTL-SDR.
    
    Args:
        duration_sec: Capture duration in seconds
        output_file: Output filename for raw IQ data
    
    Returns:
        Path to captured file
    """
    num_samples = int(duration_sec * SAMPLE_RATE)
    
    cmd = [
        "rtl_sdr",
        "-f", str(FREQ_HZ),
        "-s", str(SAMPLE_RATE),
        "-g", str(GAIN),
        "-n", str(num_samples),
        output_file
    ]
    
    print(f"Capturing {duration_sec}s at {FREQ_HZ/1e6:.3f} MHz...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    
    print(f"Captured {os.path.getsize(output_file)} bytes to {output_file}")
    return output_file

def load_iq(filename):
    """Load IQ data from rtl_sdr capture file.
    
    RTL-SDR outputs interleaved unsigned 8-bit I/Q samples.
    Convert to complex float centered at zero.
    """
    raw = np.fromfile(filename, dtype=np.uint8)
    # Convert to float, center at zero
    iq = raw.astype(np.float32) - 127.5
    # Interleave to complex
    i = iq[0::2]
    q = iq[1::2]
    return i + 1j * q

if __name__ == "__main__":
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    capture(duration)
