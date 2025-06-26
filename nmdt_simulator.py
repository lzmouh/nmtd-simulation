
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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

st.set_page_config(page_title="NMTD Simulator", layout="wide")
st.markdown("<h1 style='text-align: center; color: navy;'>🔍 Non-Metallic Tubular Defectoscope (NMDT) Simulator</h1>", unsafe_allow_html=True)

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

        # Plotly time-domain interactive plot
        time_fig = go.Figure()
        if show_perfect and superpose:
            time_fig.add_trace(go.Scatter(x=t_p, y=s_p, mode='lines', name='Perfect Pipe',
                                          line=dict(dash='dash', color='green')))
        time_fig.add_trace(go.Scatter(x=t_d, y=s_d, mode='lines', name='Defective Pipe',
                                      line=dict(color='red')))
        for t, a in zip(e_d, a_d):
            time_fig.add_vline(x=t, line_width=1, line_dash="dot", line_color="gray")
        time_fig.add_vline(x=TT_d, line_width=2, line_dash="dash", line_color="blue",
                           annotation_text=f"TT_fluid = {TT_d:.2f} µs", annotation_position="top left")
        time_fig.update_layout(title="🟢 A-Scan (Time Domain)", xaxis_title="Time (µs)",
                               yaxis_title="Amplitude", hovermode="x unified")
        st.plotly_chart(time_fig, use_container_width=True)

        # FFT plot
        dt = 1e-6
        freqs = fftfreq(len(t_d), dt)
        fft_d = np.abs(fft(s_d))
        fft_p = np.abs(fft(s_p)) if show_perfect and superpose else None

        freq_fig = go.Figure()
        if fft_p is not None:
            freq_fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_p[:len(freqs)//2],
                                          mode='lines', name='Perfect Pipe', line=dict(dash='dash')))
        freq_fig.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_d[:len(freqs)//2],
                                      mode='lines', name='Defective Pipe', line=dict(color='red')))
        freq_fig.update_layout(title="🔵 Frequency Domain (Zoomed 100 kHz – 2 MHz)",
                               xaxis_title="Frequency (Hz)", yaxis_title="Magnitude",
                               xaxis_range=[1e5, 2e6])
        st.plotly_chart(freq_fig, use_container_width=True)

        st.success(f"Simulation complete. TT_fluid = {TT_d:.2f} µs")

with tab3:
    st.header("📐 Pipe Geometry and Tool Placement")
    fig, ax = plt.subplots(figsize=(10, 2))
    total_thickness = sum([t for _, t, _ in layer_data])
    x_start = 0
    cmap = plt.get_cmap('tab20')
    ax.barh(y=0.5, width=DEFAULT_GAP_INCH, height=1, left=x_start, color='skyblue', edgecolor='black')
    ax.text(x_start + DEFAULT_GAP_INCH/2, 0.5, f"Fluid\nZ_fluid={Z_fluid:.2f}", ha='center', va='center')
    x_start += DEFAULT_GAP_INCH
    for i, (label, thickness, Z) in enumerate(layer_data):
        ax.barh(y=0.5, width=thickness, height=1, left=x_start, color=cmap(i), edgecolor='black')
        ax.text(x_start + thickness/2, 0.5, f"{label}\nZ={Z:.2f}\n{thickness:.2f} in", ha='center', va='center')
        x_start += thickness
    ax.set_xlim(0, x_start)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"Pipe Thickness View (Rotated) | Total Thickness: {total_thickness:.2f} in", fontsize=11)
    st.pyplot(fig)

    st.subheader("🛠 Tool Inside Pipe (Concept)")
    tool_fig, tool_ax = plt.subplots(figsize=(4, 8))
    pipe_radius = 1.5
    tool_radius = 0.5
    pipe_height = 6
    tool_ax.add_patch(plt.Circle((0, pipe_height / 2), pipe_radius, fill=False, linewidth=2))
    tool_ax.add_patch(plt.Rectangle((-tool_radius, 2), 2 * tool_radius, 2, color='gray'))
    tool_ax.plot([-pipe_radius, -tool_radius], [3, 3], color='red', linewidth=3)
    tool_ax.plot([pipe_radius, tool_radius], [3, 3], color='red', linewidth=3)
    tool_ax.text(0, 3.5, "Pads", ha='center')
    tool_ax.set_xlim(-2, 2)
    tool_ax.set_ylim(0, pipe_height)
    tool_ax.set_aspect('equal')
    tool_ax.axis('off')
    st.pyplot(tool_fig)
