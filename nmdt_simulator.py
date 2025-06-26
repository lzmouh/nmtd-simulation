import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

INCH_TO_METER = 0.0254
DEFAULT_GAP_INCH = 0.1
C_WATER = 1480  # m/s

fluid_impedance_db = {
    "Water": 1.48,
    "Oil": 1.20,
    "Water-based Mud (WBM)": 1.60,
    "Oil-based Mud (OBM)": 1.30,
    "Diesel": 1.25,
    "Other": None
}

st.set_page_config(page_title="NMDT Simulator", layout="wide")
st.title("🔍 Non-Metallic Tubular Defectoscope (NMDT) Simulator")

tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📊 Results", "📐 Drawing"])

with tab1:
    st.header("Pipe and Fluid Configuration")
    fluid = st.selectbox("Select Fluid Type", list(fluid_impedance_db.keys()))
    if fluid == "Other":
        fluid_density = st.number_input("Enter Fluid Density (g/cc)", min_value=0.5, max_value=2.5, value=1.0)
        Z_fluid = fluid_density * 1.48
        fluid_velocity = C_WATER
    else:
        Z_fluid = fluid_impedance_db[fluid]
        fluid_velocity = C_WATER
    st.write(f"**Acoustic Impedance (Z_fluid)** = {Z_fluid:.2f} MRayl")

    num_layers = st.slider("Number of Pipe Layers", 1, 10, 5)
    layer_data = []
    for i in range(num_layers):
        col1, col2 = st.columns(2)
        with col1:
            thickness = st.number_input(f"Layer {i+1} Thickness (in)", value=0.2, key=f"t_{i}")
        with col2:
            impedance = st.number_input(f"Layer {i+1} Acoustic Impedance Z (MRayl)", value=2.5, key=f"z_{i}")
        layer_data.append((f"Layer {i+1}", thickness, impedance))

    st.subheader("Defect Configuration")
    defect_type = st.selectbox("Defect Type", ["None", "Delamination", "Crack"])
    defect_layer = st.slider("Select Defect Layer", 1, num_layers, 2)

with tab2:
    st.header("Simulation and Results")
    show_perfect = st.checkbox("Show Perfect Pipe Response", True)
    superpose = st.checkbox("Superpose Defect and Perfect", True)

    def simulate(layer_data, velocity, defect_type=None, defect_layer=None):
        times, amps = [], []
        depth = DEFAULT_GAP_INCH * INCH_TO_METER
        TT_fluid = 2 * depth / fluid_velocity * 1e6
        times.append(TT_fluid)
        amps.append(1.0)
        for i, (_, t_inch, _) in enumerate(layer_data):
            thickness_m = t_inch * INCH_TO_METER
            depth += thickness_m
            t = 2 * depth / velocity * 1e6
            a = 1.0
            if defect_type == "Delamination" and i == defect_layer:
                t += 0.6
                a *= 0.6
            elif defect_type == "Crack" and i == defect_layer:
                a *= 0.5
            times.append(t)
            amps.append(a)
        t_axis = np.linspace(0, max(times) + 2, 3000)
        signal = np.zeros_like(t_axis)
        for t, a in zip(times, amps):
            signal += a * np.exp(-((t_axis - t)**2) / (2 * (0.05**2)))
        signal += 0.01 * np.random.randn(len(signal))
        return t_axis, signal, times, amps, TT_fluid

    if st.button("🔁 Run Simulation"):
        v_nominal = 2000
        t_p, s_p, e_p, a_p, TT_p = simulate(layer_data, v_nominal)
        t_d, s_d, e_d, a_d, TT_d = simulate(layer_data, v_nominal,
                                            defect_type if defect_type != "None" else None,
                                            defect_layer - 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        if show_perfect and superpose:
            ax.plot(t_p, s_p, '--', alpha=0.6, label="Perfect Pipe")
        ax.plot(t_d, s_d, color='red', label="Defective Pipe")
        for t, a in zip(e_d, a_d):
            ax.axvline(t, color='gray', linestyle=':', alpha=0.4)
            ax.annotate(f"{t:.2f} µs\nA={a:.2f}", xy=(t, a), xytext=(t + 0.5, a),
                        arrowprops=dict(arrowstyle="->", lw=0.5))
        ax.axvline(TT_d, color='blue', linestyle='--', label=f"TT_fluid: {TT_d:.2f} µs")
        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Amplitude")
        ax.set_title("🟢 A-Scan (Time Domain)")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # FFT
        dt = 1e-6
        freqs = fftfreq(len(t_d), dt)
        fft_d = np.abs(fft(s_d))
        fft_p = np.abs(fft(s_p)) if show_perfect and superpose else None

        fig2, ax2 = plt.subplots(figsize=(10, 3))
        if fft_p is not None:
            ax2.plot(freqs[:len(freqs)//2], fft_p[:len(freqs)//2], '--', alpha=0.6, label="Perfect Pipe")
        ax2.plot(freqs[:len(freqs)//2], fft_d[:len(freqs)//2], color='red', label="Defective Pipe")
        ax2.set_xlim(0.1e6, 2e6)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Magnitude")
        ax2.set_title("🔵 Frequency Domain (Zoomed 100 kHz – 2 MHz)")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        st.success(f"Simulation complete. TT_fluid = {TT_d:.2f} µs")

with tab3:
    st.header("📐 Pipe Cross Section (Sensor View)")
    fig, ax = plt.subplots(figsize=(6, 6))
    total_thickness = sum([t for _, t, _ in layer_data])
    y_start = 0
    cmap = plt.get_cmap('tab20')
    ax.bar(x=0.5, height=DEFAULT_GAP_INCH, width=1, bottom=y_start, color='skyblue', edgecolor='black')
    ax.text(0.5, y_start + DEFAULT_GAP_INCH/2, f"Fluid\nZ_fluid={Z_fluid:.2f}", ha='center', va='center')
    y_start += DEFAULT_GAP_INCH
    for i, (label, thickness, Z) in enumerate(layer_data):
        ax.bar(x=0.5, height=thickness, width=1, bottom=y_start, color=cmap(i), edgecolor='black')
        ax.text(0.5, y_start + thickness/2, f"{label}\nZ={Z:.2f} MRayl\n{thickness:.2f} in",
                ha='center', va='center', fontsize=9)
        y_start += thickness
    ax.set_ylim(0, y_start + 0.5)
    ax.set_xlim(0, 1)
    ax.axis('off')
    ax.set_title(f"Total Pipe Thickness: {total_thickness:.2f} in", fontsize=11)
    st.pyplot(fig)
