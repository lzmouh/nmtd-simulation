import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Arc, Rectangle, Circle
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

# App config
st.set_page_config(page_title="NMTD Simulator", layout="wide")
st.sidebar.title("📁 Menu")
page = st.sidebar.radio("Navigation", ["Simulator", "Visualization", "About"])

if "layer_data" not in st.session_state:
    st.session_state["layer_data"] = []
if "Z_fluid" not in st.session_state:
    st.session_state["Z_fluid"] = 1.48

# -------------------- SIMULATOR --------------------
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
        t_p, s_p, e_p, a_p, TT_p = simulate(layer_data, v_nominal)
        t_d, s_d, e_d, a_d, TT_d = simulate(layer_data, v_nominal,
                                            defect_type if defect_type != "None" else None,
                                            defect_layer - 1)

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

# -------------------- VISUALIZATION --------------------
elif page == "Visualization":
    st.title("📐 Tool and Pipe Visualization")

    layer_data = st.session_state.get("layer_data", [])
    Z_fluid = st.session_state.get("Z_fluid", 1.48)
    num_layers = len(layer_data)
    defect_type = st.selectbox("Visualize Defect", ["None", "Delamination", "Crack"])
    defect_layer = st.slider("Defect Layer to Highlight", 1, num_layers, 2) - 1

    def draw_nmted_visualization(layer_data, Z_fluid, defect_type, defect_layer):
        cmap = plt.get_cmap("tab20")
        pipe_thickness = sum([t for _, t, _ in layer_data])
        fluid_gap = 0.1
        tool_diameter = 3
        pipe_id = 6
        pipe_od = pipe_id + 2 * pipe_thickness

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 3])
        ax1 = fig.add_subplot(gs[0])

        # Vertical cross-section
        x0 = 0.1
        y = 0.1
        width = 0.8
        height_unit = 1.0

        ax1.add_patch(Rectangle((x0, y), width, height_unit, color='gray'))
        ax1.text(x0 + width/2, y + height_unit/2, "Sensor", ha='center', va='center', color='white')
        y += height_unit

        ax1.add_patch(Rectangle((x0, y), width, fluid_gap*4, color='skyblue'))
        ax1.text(x0 + width/2, y + (fluid_gap*2), f"Fluid\nZ={Z_fluid:.2f}", ha='center', va='center')
        y += fluid_gap*4

        for i, (label, thickness, Z) in enumerate(layer_data):
            height = thickness * 4
            color = cmap(i)
            ax1.add_patch(Rectangle((x0, y), width, height, color=color, ec='black'))
            ax1.text(x0 + width/2, y + height/2, f"{label}\nZ={Z:.2f}\n{thickness:.2f} in", ha='center', va='center')
            if defect_type != "None" and (i == defect_layer):
                ax1.plot([x0, x0 + width], [y + height / 2]*2, 'r--')
                ax1.text(x0 + width + 0.05, y + height / 2, f"⛔ {defect_type}", color='red', fontsize=9)
            y += height

        ax1.set_xlim(0, 1.5)
        ax1.set_ylim(0, y + 1)
        ax1.axis('off')
        ax1.set_title("Vertical Cross-Section")

        # Top view
        ax2 = fig.add_subplot(gs[1])
        pipe_radius = pipe_id / 2
        tool_radius = tool_diameter / 2
        radius = pipe_radius
        for i, (label, thickness, Z) in enumerate(layer_data[::-1]):
            r = radius + thickness
            ax2.add_patch(Circle((0, 0), r, color=cmap(len(layer_data)-1 - i), ec='black'))
            radius = r
        ax2.add_patch(Circle((0, 0), pipe_radius, color='skyblue', ec='black'))
        ax2.add_patch(Circle((0, 0), tool_radius, color='gray'))

        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            x0 = tool_radius * np.cos(rad)
            y0 = tool_radius * np.sin(rad)
            x1 = pipe_radius * np.cos(rad)
            y1 = pipe_radius * np.sin(rad)
            ax2.plot([x0, x1], [y0, y1], color='red', lw=4)
            arc = Arc((0, 0), 2*pipe_radius, 2*pipe_radius, angle=0,
                      theta1=angle-22.5, theta2=angle+22.5, color='red', lw=6)
            ax2.add_patch(arc)

        ax2.set_aspect('equal')
        ax2.set_xlim(-radius-1, radius+1)
        ax2.set_ylim(-radius-1, radius+1)
        ax2.axis('off')
        ax2.set_title("Top View: Tool in Pipe")
        return fig

    fig = draw_nmted_visualization(layer_data, Z_fluid, defect_type, defect_layer)
    st.pyplot(fig)

# -------------------- ABOUT --------------------
elif page == "About":
    st.title("ℹ️ About the NMTD Simulator")
    st.markdown("""
The **Non-Metallic Tubular Defectoscope (NMTD)** simulator models ultrasonic evaluation
of multilayer GRE, RTP, HDPE pipes using pad-coupled sensors and signal interpretation.

It enables:
- Thickness measurement
- Delamination & crack detection
- Visualization of tool deployment geometry

Technologies used: `Streamlit`, `NumPy`, `Matplotlib`, `Plotly`, `SciPy`.
""")
