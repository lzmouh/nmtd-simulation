import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from matplotlib.patches import Arc

# Constants
INCH_TO_METER = 0.0254
DEFAULT_GAP_INCH = 0.1
C_WATER = 1480  # m/s (assumed similar across fluids)

# Fluid acoustic impedance database
fluid_impedance_db = {
    "Water": 1.48,
    "Oil": 1.20,
    "Water-based Mud (WBM)": 1.60,
    "Oil-based Mud (OBM)": 1.30,
    "Diesel": 1.25,
    "Other": None
}

# Streamlit setup
st.set_page_config(page_title="NMTD Simulator", layout="wide")
st.sidebar.title("📁 Menu")
page = st.sidebar.radio("Navigation", ["Simulator", "Visualization", "About"])

# Shared session memory
if "layer_data" not in st.session_state:
    st.session_state["layer_data"] = []
if "Z_fluid" not in st.session_state:
    st.session_state["Z_fluid"] = 1.48  # default to water

# SIMULATOR PAGE
if page == "Simulator":
    st.title("🔍 NMTD Ultrasonic Response Simulator")

    col1, col2 = st.columns(2)
    with col1:
        fluid = st.selectbox("Select Borehole Fluid", list(fluid_impedance_db.keys()))
        if fluid == "Other":
            fluid_density = st.number_input("Fluid Density (g/cc)", 0.5, 2.5, 1.0)
            Z_fluid = fluid_density * 1.48
        else:
            Z_fluid = fluid_impedance_db[fluid]
        st.session_state["Z_fluid"] = Z_fluid
        st.write(f"**Z_fluid** = {Z_fluid:.2f} MRayl")

    with col2:
        num_layers = st.slider("Number of Layers", 1, 10, 5)
        layer_data = []
        for i in range(num_layers):
            t = st.number_input(f"Layer {i+1} Thickness (in)", value=0.2, key=f"t{i}")
            z = st.number_input(f"Layer {i+1} Impedance (MRayl)", value=2.5, key=f"z{i}")
            layer_data.append((f"Layer {i+1}", t, z))
        st.session_state["layer_data"] = layer_data

    st.subheader("📌 Defect Settings")
    defect_type = st.selectbox("Defect Type", ["None", "Delamination", "Crack"])
    defect_layer = st.slider("Defect Layer Index", 1, num_layers, 2)

    st.subheader("📈 Display Options")
    show_perfect = st.checkbox("Show Perfect Pipe", True)
    superpose = st.checkbox("Superpose Perfect and Defect", True)

    def simulate(layer_data, velocity, defect_type=None, defect_layer=None):
        times, amps = [], []
        depth = DEFAULT_GAP_INCH * INCH_TO_METER
        TT_fluid = 2 * depth / velocity * 1e6
        times.append(TT_fluid)
        amps.append(1.0)
        for i, (_, t_in, _) in enumerate(layer_data):
            depth += t_in * INCH_TO_METER
            t = 2 * depth / velocity * 1e6
            a = 1.0
            if defect_type == "Delamination" and i == defect_layer:
                t += 0.6
                a *= 0.6
            elif defect_type == "Crack" and i == defect_layer:
                a *= 0.5
            times.append(t)
            amps.append(a)

        t_axis = np.linspace(0, max(times)+2, 3000)
        signal = np.zeros_like(t_axis)
        for t, a in zip(times, amps):
            signal += a * np.exp(-((t_axis - t)**2)/(2*(0.05**2)))
        signal += 0.01 * np.random.randn(len(signal))
        return t_axis, signal, times, amps, TT_fluid

    if st.button("▶️ Run Simulation"):
        v_nominal = 2000
        layer_data = st.session_state["layer_data"]
        t_p, s_p, e_p, a_p, TT_p = simulate(layer_data, v_nominal)
        t_d, s_d, e_d, a_d, TT_d = simulate(layer_data, v_nominal,
                                            defect_type if defect_type != "None" else None,
                                            defect_layer - 1)

        # Time Domain Plot
        fig = go.Figure()
        if show_perfect and superpose:
            fig.add_trace(go.Scatter(x=t_p, y=s_p, name="Perfect Pipe", line=dict(dash='dash', color='green')))
        fig.add_trace(go.Scatter(x=t_d, y=s_d, name="Defective Pipe", line=dict(color='red')))
        for t, a in zip(e_d, a_d):
            fig.add_vline(x=t, line_dash="dot", line_color="gray")
        fig.add_vline(x=TT_d, line_dash="dash", line_color="blue",
                      annotation_text=f"TT_fluid={TT_d:.2f} µs")
        fig.update_layout(title="🟢 A-Scan (Time Domain)", xaxis_title="Time (µs)",
                          yaxis_title="Amplitude", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Frequency Domain Plot
        freqs = fftfreq(len(t_d), 1e-6)
        fft_d = np.abs(fft(s_d))
        fft_p = np.abs(fft(s_p)) if show_perfect and superpose else None
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_d[:len(freqs)//2], name="Defective Pipe", line=dict(color='red')))
        if fft_p is not None:
            fig2.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_p[:len(freqs)//2], name="Perfect Pipe", line=dict(dash='dash')))
        fig2.update_layout(title="🔵 Frequency Domain (Autoscaled)",
                           xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
        st.plotly_chart(fig2, use_container_width=True)

        st.success(f"Simulation complete. TT_fluid = {TT_d:.2f} µs")

# VISUALIZATION TAB
elif page == "Visualization":
    st.title("📐 Tool and Pipe Visualization")

    layer_data = st.session_state.get("layer_data", [])
    Z_fluid = st.session_state.get("Z_fluid", 1.48)
    num_layers = len(layer_data)

    fig, ax = plt.subplots(figsize=(6, 8))
    pipe_radius = 1.5
    pipe_length = 6
    tool_radius = 0.5
    arm_len = 1.0
    pad_len = 0.2
    layer_thickness = pipe_radius / max(num_layers, 1)
    cmap = plt.get_cmap("tab20")

    for i, (_, _, _) in enumerate(layer_data):
        r1 = pipe_radius - i*layer_thickness
        circle = plt.Circle((0, pipe_length/2), r1, fill=True, color=cmap(i), ec='black', lw=0.5)
        ax.add_patch(circle)

    ax.add_patch(plt.Circle((0, pipe_length/2), tool_radius, color='gray'))

    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        x1 = tool_radius * np.cos(rad)
        y1 = pipe_length/2 + tool_radius * np.sin(rad)
        x2 = (tool_radius + arm_len) * np.cos(rad)
        y2 = pipe_length/2 + (tool_radius + arm_len) * np.sin(rad)
        ax.plot([x1, x2], [y1, y2], color='red', lw=2)
        ax.add_patch(Arc((x2, y2), width=pad_len, height=pad_len, theta1=0, theta2=180, color='red'))

    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, pipe_length)
    ax.axis('off')
    ax.set_title("Tool Inside Non-Metallic Pipe")
    st.pyplot(fig)

# ABOUT PAGE
elif page == "About":
    st.title("ℹ️ About the NMTD Simulator")
    st.markdown("""
The **Non-Metallic Tubular Defectoscope (NMTD)** simulator is an interactive tool to study
ultrasonic signal behavior in **multi-layer non-metallic pipes** like GRE, RTP, and HDPE.

### 🔧 Purpose
- Measure layer thickness
- Detect delaminations and cracks
- Visualize internal tool-pipe interaction

### ⚙️ How It Works
- An ultrasonic pulse travels through a 0.1 in fluid gap and multiple non-metallic layers
- Reflections at interfaces are simulated
- Time-domain and frequency-domain responses are analyzed

Built using `Streamlit`, `NumPy`, `SciPy`, `Plotly`, and `Matplotlib`.
""")
