import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Arc, Rectangle, Circle
from scipy.fft import fft, fftfreq
import json
import pandas as pd

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

DEFAULT_CONFIG = {
    "fluid": "Water",
    "Z_fluid": 1.48,
    "num_layers": 5,
    "layer_data": [["Layer 1", 0.2, 2.5]] * 5,
    "defect_type": "None",
    "defect_layer": 2
}

# ---------- SETUP ----------
st.set_page_config(page_title="NMTD Simulator", layout="wide")
st.sidebar.title("📁 Menu")
page = st.sidebar.radio("Navigation", ["Simulator", "Plots", "Visualization", "About"])

# ---------- STATE ----------
if "config" not in st.session_state:
    if "config" not in st.session_state:
    st.session_state["config"] = json.loads(json.dumps(DEFAULT_CONFIG)) 
    
config = st.session_state["config"]

# -------------------- SIMULATOR --------------------
if page == "Simulator":
    st.title("🔍 NMTD Ultrasonic Response Simulator")

    col1, col2 = st.columns(2)

    # --- Fluid selection ---
    with col1:
        config["fluid"] = st.selectbox(
            "Select Borehole Fluid", list(fluid_impedance_db.keys()),
            index=list(fluid_impedance_db.keys()).index(config["fluid"])
        )
        if config["fluid"] == "Other":
            fluid_density = st.number_input("Fluid Density (g/cc)", 0.5, 2.5, 1.0)
            config["Z_fluid"] = fluid_density * 1.48
        else:
            config["Z_fluid"] = fluid_impedance_db[config["fluid"]]
        st.write(f"**Z_fluid = {config['Z_fluid']:.2f} MRayl**")

    # --- Number of layers + total thickness display ---
    with col2:
        config["num_layers"] = st.slider("Number of Layers", 1, 10, config["num_layers"])

        # Ensure layer_data list is correct length
        config["layer_data"] = config["layer_data"][:config["num_layers"]]
        while len(config["layer_data"]) < config["num_layers"]:
            config["layer_data"].append(
                [f"Layer {len(config['layer_data'])+1}", 0.2, 2.5]
            )

    # --- Layers configuration ---
    st.markdown("### 📦 Layers Configuration")
    for i in range(config["num_layers"]):
        c1, c2 = st.columns(2)
        with c1:
            config["layer_data"][i][1] = st.number_input(
                f"Layer {i+1} Thickness (in)",
                min_value=0.01, max_value=1.0,
                value=config["layer_data"][i][1],
                key=f"t_{i}"
            )
        with c2:
            config["layer_data"][i][2] = st.number_input(
                f"Layer {i+1} Impedance (MRayl)",
                min_value=1.0, max_value=5.0,
                value=config["layer_data"][i][2],
                key=f"z_{i}"
            )

        # Total thickness is calculated from current values
        total_thickness = sum([layer[1] for layer in config["layer_data"]])
        st.write(f"**📏 Total Pipe Thickness: `{total_thickness:.2f}` inches**")

    
    # --- Defect Settings ---
    st.subheader("📌 Defect Settings")
    c1, c2 = st.columns(2)
    with c1:
        config["defect_type"] = st.selectbox(
            "Defect Type", ["None", "Delamination", "Crack"],
            index=["None", "Delamination", "Crack"].index(config["defect_type"])
        )
    with c2:
        config["defect_layer"] = st.slider(
            "Defect Layer Index", 1, config["num_layers"], config["defect_layer"]
        )

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

    layer_data = config["layer_data"]
    Z_fluid = config["Z_fluid"]
    defect_type = config["defect_type"]
    defect_layer = config["defect_layer"] - 1

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
        if not layer_data:
            st.error("No data! Please configure the pipe in the Simulator tab.")
        else:
            v_nominal = 2000
            sim_defect_layer = defect_layer - 1 if defect_type != "None" else None
            t_p, s_p, e_p, a_p, TT_p = simulate(layer_data, v_nominal)
            t_d, s_d, e_d, a_d, TT_d = simulate(layer_data, v_nominal,
                                                defect_type if defect_type != "None" else None,
                                                sim_defect_layer)

            # Plot Time Domain
            fig = go.Figure()
            if show_perfect and superpose:
                fig.add_trace(go.Scatter(x=t_p, y=s_p, name="Perfect Pipe", line=dict(dash='dash', color='green')))
            fig.add_trace(go.Scatter(x=t_d, y=s_d, name="Defective Pipe", line=dict(color='red')))
            for t in e_d:
                fig.add_vline(x=t, line_dash="dot", line_color="gray")
            fig.add_vline(x=TT_d, line_dash="dash", line_color="blue",
                          annotation_text=f"TT_fluid={TT_d:.2f} µs")
            fig.update_layout(title="🟢 A-Scan (Time Domain)", xaxis_title="Time (µs)",
                              yaxis_title="Amplitude", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Frequency Domain
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

    layer_data = config["layer_data"]
    Z_fluid = config["Z_fluid"]
    defect_type = config["defect_type"]
    defect_layer = config["defect_layer"] - 1
    
    num_layers = len(layer_data)
    
    def draw_nmted_visualization(layer_data, Z_fluid,
                              defect_type=None, defect_layer=None):
        """
        layer_data: list of tuples (label, thickness_in_inches, impedance)
        Z_fluid: fluid acoustic impedance (MRayl)
        defect_type: "Crack" or "Delamination" or None
        defect_layer: zero-based index of layer for defect
        """
        cmap = plt.get_cmap("tab20")
        fig = plt.figure(figsize=(14,10))
        gs = fig.add_gridspec(2,1, height_ratios=[2,3])
    
        # --- 1) Horizontal Cross-Section ---
        ax1 = fig.add_subplot(gs[0])
        y = 0.2
        H = 0.6  # uniform height
        x = 0.1
    
        # Tool body
        W_tool = 0.25
        ax1.add_patch(Rectangle((x,y), W_tool, H, color='gray'))
        ax1.text(x+W_tool/2, y+H+0.05, "Tool Body", ha='center')
        x += W_tool
    
        # Fluid section with embedded arm
        W_arm = 0.2
        ax1.add_patch(Rectangle((x,y), W_arm, H, color='skyblue'))
        # Arm as black strip centered vertically
        ax1.add_patch(Rectangle((x, y+H/2-0.05), W_arm, 0.1, color='black'))
        ax1.text(x+W_arm/2, y+H+0.05, "Fluid + Arm", ha='center')
        x += W_arm
    
        # Sensor
        W_sensor = 0.15
        ax1.add_patch(Rectangle((x,y), W_sensor, H, color='red'))
        ax1.text(x+W_sensor/2, y+H+0.05, "Sensor", ha='center')
        x += W_sensor
    
        # Fluid gap
        W_gap = 0.1
        ax1.add_patch(Rectangle((x,y), W_gap, H, color='skyblue'))
        ax1.text(x+W_gap/2, y+H+0.05, f"Gap\nZ={Z_fluid:.2f}", ha='center')
        x += W_gap
    
        # Pipe layers
        cumulative_x = x
        for i, (label, t, Z) in enumerate(layer_data):
            W = t
            color = cmap(i)
            # draw layer
            ax1.add_patch(Rectangle((cumulative_x, y), W, H, color=color, ec='black'))
            ax1.text(cumulative_x+W/2, y+H+0.05,
                     f"Layer {i+1}\nZ={Z:.2f}\n{t:.2f}\"",
                     ha='center', fontsize=7)
            # delamination: vertical strip at boundary before this layer
            if defect_type=="Delamination" and i==defect_layer:
                ax1.add_patch(Rectangle((cumulative_x-0.01, y), 0.02, H,
                                        color='white', ec='red', lw=2))
                ax1.text(cumulative_x, y+H+0.05, "Delam.", color='red', fontsize=8, ha='left')
            # crack: horizontal line across layer
            if defect_type=="Crack" and i==defect_layer:
                ax1.plot([cumulative_x, cumulative_x+W],
                         [y+H/2, y+H/2], 'k--', lw=2)
                ax1.text(cumulative_x+W/2, y-0.1, "Crack",
                         ha='center', color='black', fontsize=8)
            cumulative_x += W
    
        ax1.set_xlim(0, cumulative_x+0.2)
        ax1.set_ylim(0, y+H+0.3)
        ax1.axis('off')
        ax1.set_title("Cross-Section: Tool → Arm → Sensor → Gap → Layers")
    
        # --- 2) Top View ---
        ax2 = fig.add_subplot(gs[1])
        pipe_id = 6.0
        tool_d = 3.0
        pad_gap = 0.1
        pad_span = 45  # degrees
    
        r_inner = pipe_id/2
        r = r_inner
        # draw concentric layer rings
        for i, (_, t, _) in enumerate(layer_data):
            r += t
            ax2.add_patch(Circle((0,0), r, color=cmap(i), ec='black'))
    
        # fluid ring
        ax2.add_patch(Circle((0,0), r_inner, color='skyblue', ec='black'))
        # tool
        tool_r = tool_d/2
        ax2.add_patch(Circle((0,0), tool_r, color='gray', ec='black'))
    
        # arms + pads
        for ang in [0,90,180,270]:
            rad = np.deg2rad(ang)
            x0,y0 = tool_r*np.cos(rad), tool_r*np.sin(rad)
            x1,y1 = (r_inner-pad_gap)*np.cos(rad), (r_inner-pad_gap)*np.sin(rad)
            ax2.plot([x0,x1],[y0,y1], 'red', lw=3)
            ax2.add_patch(Arc((0,0),
                              2*(r_inner-pad_gap),
                              2*(r_inner-pad_gap),
                              theta1=ang-pad_span/2,
                              theta2=ang+pad_span/2,
                              color='red', lw=6))
    
        # legend arrows
        # tool
        ax2.annotate("Tool Body", xy=(0,0), xytext=(tool_r+1,0),
                     arrowprops=dict(arrowstyle="->"))
        # fluid
        ax2.annotate("Fluid Gap & Pipe ID", xy=(r_inner,0), xytext=(r_inner+1,1),
                     arrowprops=dict(arrowstyle="->"))
        # layers
        for i, (_, t, _) in enumerate(layer_data):
            radius = r_inner + sum(th for _,th,_ in layer_data[:i+1])
            color = cmap(i)
            ax2.annotate(f"Layer {i+1}", xy=(radius,0),
                         xytext=(radius+0.5,-i-1),
                         color=color,
                         arrowprops=dict(arrowstyle="->", color=color))
    
        ax2.set_aspect('equal')
        ax2.set_xlim(-r-1, r+1)
        ax2.set_ylim(-r-1, r+1)
        ax2.axis('off')
        ax2.set_title("Top View: Tool & Pads inside Multilayer Pipe")
    
        plt.tight_layout()
        return fig

    fig = draw_nmted_visualization(layer_data, Z_fluid, defect_type, defect_layer)
    st.pyplot(fig)

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
