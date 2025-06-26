
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Constants
INCH_TO_METER = 0.0254
DEFAULT_GAP_INCH = 0.1
C_WATER = 1480  # m/s

# Acoustic impedance lookup (MRayl)
fluid_impedance_db = {
    "Water": 1.48,
    "Oil": 1.20,
    "Water-based Mud (WBM)": 1.60,
    "Oil-based Mud (OBM)": 1.30,
    "Diesel": 1.25,
    "Other": None
}

st.title("NMDT Ultrasonic Response Simulator")

# --- Fluid Input ---
fluid = st.selectbox("Select Fluid Type", list(fluid_impedance_db.keys()))
if fluid == "Other":
    fluid_density = st.number_input("Enter Fluid Density (g/cc)", min_value=0.5, max_value=2.5, value=1.0)
    fluid_impedance = fluid_density * 1.48
    fluid_velocity = C_WATER  # assume similar to water
else:
    fluid_impedance = fluid_impedance_db[fluid]
    fluid_velocity = C_WATER

st.write(f"**Acoustic Impedance of Fluid:** {fluid_impedance:.2f} MRayl")

# --- Pipe Configuration ---
st.header("Pipe Layers Configuration")
num_layers = st.slider("Number of Layers", min_value=1, max_value=10, value=5)
layer_data = []
for i in range(num_layers):
    col1, col2 = st.columns(2)
    with col1:
        thickness = st.number_input(f"Layer {i+1} Thickness (in)", value=0.2, key=f"t_{i}")
    with col2:
        impedance = st.number_input(f"Layer {i+1} Impedance (MRayl)", value=2.5, key=f"z_{i}")
    layer_data.append((thickness, impedance))

# --- Defect Settings ---
st.header("Defect Settings")
defect_type = st.selectbox("Defect Type", ["None", "Delamination", "Crack"])
defect_layer = st.slider("Layer with Defect", min_value=1, max_value=num_layers, value=2)

# --- Display Options ---
st.header("Display Options")
show_perfect = st.checkbox("Display Perfect Pipe Response", value=True)
superpose = st.checkbox("Superpose Defect and Perfect Responses", value=True)

# --- Simulate Ultrasonic Signal ---
def simulate_ultrasound(layer_data, velocity, gap_inch=DEFAULT_GAP_INCH, defect_type=None, defect_layer=None):
    times = []
    amplitudes = []
    depth = gap_inch * INCH_TO_METER
    tof_fluid = 2 * depth / fluid_velocity * 1e6  # Âµs
    times.append(tof_fluid)
    amplitudes.append(1.0)

    for i, (t_inch, _) in enumerate(layer_data):
        thickness_m = t_inch * INCH_TO_METER
        depth += thickness_m
        t = 2 * depth / velocity * 1e6  # Âµs
        a = 1.0
        if defect_type == "Delamination" and i == defect_layer:
            t += 0.6
            a *= 0.6
        elif defect_type == "Crack" and i == defect_layer:
            a *= 0.5
        times.append(t)
        amplitudes.append(a)

    time_axis = np.linspace(0, max(times) + 2, 3000)
    signal = np.zeros_like(time_axis)
    for t, a in zip(times, amplitudes):
        signal += a * np.exp(-((time_axis - t)**2) / (2 * (0.05**2)))
    signal += 0.01 * np.random.randn(len(signal))
    return time_axis, signal, times, amplitudes, tof_fluid

# --- Plot Time Domain Response ---
def plot_time_domain(time_axis, signal_d, label_d, echoes_d, amps_d, tof_fluid, signal_p=None, label_p=None, echoes_p=None, amps_p=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    if signal_p is not None and show_perfect:
        ax.plot(time_axis, signal_p, label=label_p, linestyle='--', alpha=0.7)
    ax.plot(time_axis, signal_d, label=label_d, color='red')

    # Annotate echoes
    for t, a in zip(echoes_d, amps_d):
        ax.axvline(t, color='gray', linestyle=':', alpha=0.4)
        ax.annotate(f"{t:.2f} Âµs\nAmp: {a:.2f}", xy=(t, a), xytext=(t + 0.5, a), arrowprops=dict(arrowstyle="->", lw=0.5))

    ax.axvline(tof_fluid, color='blue', linestyle='--', label=f"TOF Fluid: {tof_fluid:.2f} Âµs")
    ax.set_xlabel("Time (Âµs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("A-Scan (Time Domain)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Plot Frequency Domain ---
def plot_frequency_domain(signal_d, label_d, signal_p=None, label_p=None):
    N = len(signal_d)
    dt = 1e-6  # Âµs sampling in seconds
    freqs = fftfreq(N, dt)
    fft_d = np.abs(fft(signal_d))
    fft_p = np.abs(fft(signal_p)) if signal_p is not None else None

    fig, ax = plt.subplots(figsize=(10, 3))
    if fft_p is not None and show_perfect:
        ax.plot(freqs[:N//2], fft_p[:N//2], label=label_p, linestyle='--', alpha=0.7)
    ax.plot(freqs[:N//2], fft_d[:N//2], label=label_d, color='red')
    ax.set_xlim(0, 2e6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Domain Response")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Main Simulation Trigger ---
if st.button("Simulate NMDT Response"):

    v_nominal = 2000  # m/s assumed for non-metallic layers

    # Perfect pipe
    time_p, signal_p, echoes_p, amps_p, tof_p = simulate_ultrasound(layer_data, v_nominal)

    # Defective pipe
    time_d, signal_d, echoes_d, amps_d, tof_d = simulate_ultrasound(layer_data, v_nominal, defect_type=defect_type if defect_type != "None" else None, defect_layer=defect_layer-1)

    # Plot time domain
    plot_time_domain(time_d, signal_d, "Defect Response", echoes_d, amps_d, tof_d,
                     signal_p if superpose else None, "Perfect Response" if superpose else None,
                     echoes_p if superpose else None, amps_p if superpose else None)

    # Plot frequency domain
    plot_frequency_domain(signal_d, "Defect Response", signal_p if superpose else None, "Perfect Response" if superpose else None)

    st.success(f"Simulation complete. TOF of fluid (gap) = {tof_d:.2f} Âµs")

