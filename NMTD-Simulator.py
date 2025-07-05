import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Arc, Rectangle, Circle
from scipy.fft import fft, fftfreq
import json
import pandas as pd
from matplotlib.patches import Wedge

# ---------- CONSTANTS ----------
INCH_TO_METER = 0.0254
DEFAULT_GAP_INCH = 0.1
DEFAULT_VELOCITY = 2000  # m/s

fluid_impedance_db = {
    "Water": 1.48,
    "Oil": 1.20,
    "Water-based Mud (WBM)": 1.60,
    "Oil-based Mud (OBM)": 1.30,
    "Diesel": 1.25,
    "Other": None
}

default_densities = {
    "Water": 1.0,
    "Oil": 0.85,
    "Water-based Mud (WBM)": 1.2,
    "Oil-based Mud (OBM)": 1.1,
    "Diesel": 0.82
}

DEFAULT_CONFIG = {
    "fluid": "Water",
    "Z_fluid": 1.48,
    "fluid_density": 1.0,
    "fluid_velocity": 1480,
    "num_layers": 5,
    "layer_data": [["Layer 1", 0.2, 2.5]] * 5,
    "defect_type": "None",
    "defect_layer": 2
}

# ---------- SETUP ----------
st.set_page_config(page_title="NMTD Simulator", layout="wide")
st.sidebar.title("📁 Menu")
page = st.sidebar.radio("Navigation", ["Simulator", "Plots", "Visualization", "About"])

if "config" not in st.session_state:
    st.session_state["config"] = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

config = st.session_state["config"]

# -------------------- SIMULATOR --------------------
if page == "Simulator":
    st.title("🔍 NMTD Ultrasonic Response Simulator")

    col1, col2 = st.columns(2)
    with col1:
        config["fluid"] = st.selectbox(
            "Select Borehole Fluid", list(fluid_impedance_db.keys()),
            index=list(fluid_impedance_db.keys()).index(config["fluid"])
        )
        if config["fluid"] == "Other":
            config["fluid_density"] = st.number_input("Fluid Density (g/cc)", 0.5, 2.5, 1.0)
        else:
            config["fluid_density"] = default_densities[config["fluid"]]

        config["Z_fluid"] = fluid_impedance_db[config["fluid"]] if config["fluid"] != "Other" else config["Z_fluid"]

        Z = config["Z_fluid"] * 1e6  # Rayl
        rho = config["fluid_density"] * 1000  # kg/m³
        c_fluid = Z / rho  # m/s
        config["fluid_velocity"] = c_fluid

        st.write(f"**Z_fluid = {config['Z_fluid']:.2f} MRayl**")
        st.write(f"**Fluid velocity = {c_fluid:.0f} m/s**")

    with col2:
        config["num_layers"] = st.slider("Number of Layers", 1, 10, config["num_layers"])

    config["layer_data"] = config["layer_data"][:config["num_layers"]]
    while len(config["layer_data"]) < config["num_layers"]:
        config["layer_data"].append([f"Layer {len(config['layer_data'])+1}", 0.2, 2.5])

    st.markdown("### 📦 Layers Configuration")
    for i in range(config["num_layers"]):
        c1, c2 = st.columns(2)
        with c1:
            config["layer_data"][i][1] = st.number_input(
                f"Layer {i+1} Thickness (in)", min_value=0.01, max_value=1.0,
                value=config["layer_data"][i][1], key=f"t_{i}"
            )
        with c2:
            config["layer_data"][i][2] = st.number_input(
                f"Layer {i+1} Z (MRayl)", min_value=1.0, max_value=5.0,
                value=config["layer_data"][i][2], key=f"z_{i}"
            )

    total_thickness = sum([layer[1] for layer in config["layer_data"]])
    st.markdown(f"**📏 Total Pipe Thickness: `{total_thickness:.2f}` inches**")

    st.subheader("📌 Defect Settings")
    c1, c2 = st.columns(2)
    with c1:
        config["defect_type"] = st.selectbox("Defect Type", ["None", "Delamination", "Crack"],
                                             index=["None", "Delamination", "Crack"].index(config["defect_type"]))
    with c2:
        config["defect_layer"] = st.slider("Defect Layer Index", 1, config["num_layers"], config["defect_layer"])


    # --- Save / Load / Reset ---
    st.markdown("### 💾 Save / Load / Export")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "📤 Export Config (.json)",
            data=json.dumps(config, indent=2),
            file_name="nmted_config.json",
            mime="application/json"
        )

    with col2:
        uploaded = st.file_uploader("⬆️ Load Config (.json)", type="json")
        if uploaded:
            loaded = json.load(uploaded)
            if "layer_data" in loaded:
                st.session_state["config"] = loaded
                st.success("Configuration loaded.")
                st.rerun()

    with col3:
        if st.button("🗑️ Reset to Default"):
            st.session_state["config"] = json.loads(json.dumps(DEFAULT_CONFIG))
            st.rerun()

# -------------------- PLOTS --------------------
elif page == "Plots":
    st.title("📊 Simulation Results")

    def simulate(layer_data, Z_fluid, fluid_density, defect_type=None, defect_layer=None):
        gap_m = DEFAULT_GAP_INCH * INCH_TO_METER
        Z = Z_fluid * 1e6
        rho = fluid_density * 1000
        c = Z / rho
        TT_fluid = 2 * gap_m / c * 1e6  # µs

        times = [TT_fluid]
        amps = [1.0]
        depth = gap_m

        for i, (_, t_in, _) in enumerate(layer_data):
            depth += t_in * INCH_TO_METER
            t = 2 * depth / DEFAULT_VELOCITY * 1e6
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
            signal += a * np.exp(-((t_axis - t) ** 2) / (2 * (0.05 ** 2)))
        signal += 0.01 * np.random.randn(len(signal))
        return t_axis, signal, times, amps, TT_fluid

    config = st.session_state["config"]
    layer_data = config["layer_data"]
    Z_fluid = config["Z_fluid"]
    fluid_density = config["fluid_density"]
    defect_type = config["defect_type"]
    defect_layer = config["defect_layer"]

    t_p, s_p, e_p, a_p, TT_p = simulate(layer_data, Z_fluid, fluid_density)
    t_d, s_d, e_d, a_d, TT_d = simulate(
        layer_data, Z_fluid, fluid_density,
        defect_type if defect_type != "None" else None,
        defect_layer if defect_type != "None" else None
    )

    show_perfect = st.checkbox("Show Perfect Pipe", True)
    superpose = st.checkbox("Superpose Perfect and Defect", True)

    fig = go.Figure()
    if show_perfect and superpose:
        fig.add_trace(go.Scatter(x=t_p, y=s_p, name="Perfect Pipe", line=dict(dash='dash', color='green')))
    fig.add_trace(go.Scatter(x=t_d, y=s_d, name="Defective Pipe", line=dict(color='red')))
    for t in e_d:
        fig.add_vline(x=t, line_dash="dot", line_color="gray")
    fig.add_vline(x=TT_d, line_dash="dash", line_color="blue",
                  annotation_text=f"TT_fluid={TT_d:.2f} µs")
    fig.update_layout(title="🟢 A-Scan (Time Domain)",
                      xaxis_title="Time (µs)", yaxis_title="Amplitude")
    st.plotly_chart(fig, use_container_width=True)

    freqs = fftfreq(len(t_d), 1e-6)
    fft_d = np.abs(fft(s_d))
    fft_p = np.abs(fft(s_p))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_d[:len(freqs)//2], name="Defective Pipe", line=dict(color='red')))
    if show_perfect and superpose:
        fig2.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=fft_p[:len(freqs)//2], name="Perfect Pipe", line=dict(dash='dash')))
    fig2.update_layout(title="🔵 Frequency Domain",
                       xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig2, use_container_width=True)


# -------------------- VISUALIZATION --------------------
elif page == "Visualization":
    st.title("📐 Tool and Pipe Visualization")

    layer_data = config["layer_data"]
    Z_fluid = config["Z_fluid"]
    defect_type = config["defect_type"]
    defect_layer = config["defect_layer"] - 1  # zero-based index

    cmap = plt.get_cmap("tab20")

    # --------- 1) Horizontal Cross Section (FULL WIDTH) ---------
    fig1 = plt.figure(figsize=(16, 3))
    ax1 = fig1.add_subplot(111)

    y = 0.2
    H = 0.6
    x = 0.1

    # Tool Body
    W_tool = 0.25
    ax1.add_patch(Rectangle((x, y), W_tool, H, color='gray'))
    ax1.text(x + W_tool / 2, y + H + 0.05, "Tool Body", ha='center')
    x += W_tool

    # Arm + Fluid
    W_arm = 0.2
    ax1.add_patch(Rectangle((x, y), W_arm, H, color='skyblue'))
    ax1.add_patch(Rectangle((x, y + H / 2 - 0.05), W_arm, 0.1, color='black'))
    ax1.text(x + W_arm / 2, y + H + 0.05, "Fluid + Arm", ha='center')
    x += W_arm

    # Sensor
    W_sensor = 0.15
    ax1.add_patch(Rectangle((x, y), W_sensor, H, color='red'))
    ax1.text(x + W_sensor / 2, y + H + 0.05, "Sensor", ha='center')
    x += W_sensor

    # Fluid Gap
    W_gap = 0.1
    ax1.add_patch(Rectangle((x, y), W_gap, H, color='skyblue'))
    ax1.text(x + W_gap / 2, y + H + 0.05, f"Gap\nZ={Z_fluid:.2f}", ha='center')
    x += W_gap

    # Pipe Layers
    for i, (label, t, Z) in enumerate(layer_data):
        W = t
        color = cmap(i)
        ax1.add_patch(Rectangle((x, y), W, H, color=color, ec='black'))
        ax1.text(x + W / 2, y + H + 0.05, f"Layer {i+1}\nZ={Z:.2f}\n{t:.2f}\"",
                 ha='center', fontsize=7)

        if defect_type == "Delamination" and i == defect_layer:
            ax1.add_patch(Rectangle((x - 0.01, y), 0.02, H,
                                    color='white', ec='red', lw=2))
            ax1.text(x, y + H + 0.05, "Delam.", color='red', fontsize=8, ha='left')
        elif defect_type == "Crack" and i == defect_layer:
            ax1.plot([x, x + W], [y + H / 2, y + H / 2], 'k--', lw=2)
            ax1.text(x + W / 2, y - 0.1, "Crack", ha='center', color='black', fontsize=8)
        x += W

    ax1.set_xlim(0, x + 1)
    ax1.set_ylim(0, y + H + 0.4)
    ax1.axis('off')
    ax1.set_title("Cross-Section: Tool → Arm → Sensor → Gap → Pipe Layers")

    st.pyplot(fig1, use_container_width=True)

    # --------- 2) Top View Drawing ---------
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)

    pipe_id = 6.0
    tool_d = 3.0
    pad_gap = 0.1
    pad_span = 45

    r_inner = pipe_id / 2
    tool_r = tool_d / 2
    r_current = r_inner
    layer_radii = []

    
    # Step 1: Draw pipe layers as full ring segments (Wedges)
    for i, (_, t, _) in enumerate(layer_data):
        r_outer = r_current + t
        color = cmap(i)
        ring = Wedge(center=(0, 0),
                     r=r_outer,
                     theta1=0,
                     theta2=360,
                     width=t,
                     facecolor=color,
                     edgecolor='black',
                     lw=1,
                     zorder=1)
        ax2.add_patch(ring)
        layer_radii.append((r_current, r_outer, color))
        r_current = r_outer    

    # Fluid gap ring
    ax2.add_patch(Circle((0, 0), r_inner, color='skyblue', ec='black', zorder=3))

    # Tool body
    ax2.add_patch(Circle((0, 0), tool_r, color='gray', ec='black', zorder=4))

    # Arms and pads
    for ang in [0, 90, 180, 270]:
        rad = np.deg2rad(ang)
        x0 = tool_r * np.cos(rad)
        y0 = tool_r * np.sin(rad)
        x1 = (r_inner - pad_gap) * np.cos(rad)
        y1 = (r_inner - pad_gap) * np.sin(rad)
        ax2.plot([x0, x1], [y0, y1], 'red', lw=3, zorder=5)
        ax2.add_patch(Arc((0, 0),
                          2 * (r_inner - pad_gap),
                          2 * (r_inner - pad_gap),
                          theta1=ang - pad_span / 2,
                          theta2=ang + pad_span / 2,
                          color='red', lw=6, zorder=5))

     # Optional defects
    if defect_type == "Delamination":
        r_delam = r_inner + sum(layer_data[i][1] for i in range(defect_layer))
        delam_ring = Wedge(center=(0, 0),
                           r=r_delam + 0.01,  # thin white outer ring
                           theta1=0,
                           theta2=45,
                           width=0.01,
                           facecolor='white',
                           edgecolor='red',
                           lw=0.5,
                           zorder=10)
        ax2.add_patch(delam_ring)

    elif defect_type == "Crack":
        r_start = r_inner + sum(layer_data[i][1] for i in range(defect_layer))
        r_end = r_start + layer_data[defect_layer][1]
        ang = np.deg2rad(45)
        x1 = r_start * np.cos(ang)
        y1 = r_start * np.sin(ang)
        x2 = r_end * np.cos(ang)
        y2 = r_end * np.sin(ang)
        ax2.plot([x1, x2], [y1, y2], 'black', lw=2, linestyle='--', zorder=10)
    

    # Annotations
    ax2.annotate("Tool Body", xy=(tool_r, 0), xytext=(tool_r + 1.5, 1.2),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax2.annotate("Fluid Gap", xy=(r_inner, 0), xytext=(r_inner + 1.5, -1.5),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)

    for i, (r_in, r_out, color) in enumerate(layer_radii):
        angle = 30 + i * 25
        rad = np.deg2rad(angle)
        x = (r_out - 0.1) * np.cos(rad)
        y = (r_out - 0.1) * np.sin(rad)
        xt = (r_out + 1.0) * np.cos(rad)
        yt = (r_out + 1.0) * np.sin(rad)
        ax2.annotate(f"Layer {i+1}",
                     xy=(x, y), xytext=(xt, yt),
                     color=color, fontsize=9,
                     arrowprops=dict(arrowstyle="->", color=color))

    ax2.set_aspect('equal')
    ax2.set_xlim(-r_current - 2, r_current + 2)
    ax2.set_ylim(-r_current - 2, r_current + 2)
    ax2.axis('off')
    ax2.set_title("Top View: Tool & Pads inside Multilayer Pipe")

    st.pyplot(fig2)
    

# -------------------- ABOUT --------------------
elif page == "About":
    st.title("ℹ️ About the NMTD Simulator")
    st.markdown("""

The **Non-Metallic Tubular Defectoscope (NMDT)** simulator is an interactive tool for simulating and visualizing
ultrasonic responses in **non-metallic multi-layer pipes** such as GRE, RTP, and HDPE.

### 🎯 Purpose
To evaluate:
- Layer thickness
- Delaminations
- Cracks
- Interface reflections

### 🧪 How It Works
- The ultrasonic sensor is deployed on a pad that contacts the inner wall.
- A pulse travels through a small fluid gap (0.1 in) and multi-layered pipe.
- Each interface reflects a portion of the wave.
- Time delay (TT) and amplitude help detect defects.

### 🧰 Technologies Used
- `Streamlit` for web interface
- `NumPy`, `SciPy` for signal simulation
- `Plotly` and `Matplotlib` for visualization

To customize or deploy this tool, contact your developer or AI support engineer.
""")
