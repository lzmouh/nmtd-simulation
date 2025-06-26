import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Predefined fluid acoustic impedance (MRayl)
fluid_impedance_db = {
    "Water": 1.48,
    "Oil": 1.20,
    "Water-based Mud (WBM)": 1.60,
    "Oil-based Mud (OBM)": 1.30,
    "Diesel": 1.25,
    "Other": None
}

st.title("NMDT Ultrasonic Response Simulator")

# Fluid Selection
fluid = st.selectbox("Select Fluid Type", list(fluid_impedance_db.keys()))
if fluid == "Other":
    fluid_density = st.number_input("Enter Fluid Density (g/cc)", min_value=0.5, max_value=2.5, value=1.0)
    fluid_impedance = fluid_density * 1.48  # Approximate scaling
else:
    fluid_impedance = fluid_impedance_db[fluid]

st.write(f"**Acoustic Impedance of Fluid:** {fluid_impedance:.2f} MRayl")

# Pipe Configuration
st.header("Pipe Structure")
num_layers = st.slider("Number of Layers", min_value=1, max_value=10, value=5)
layer_data = []

for i in range(num_layers):
    col1, col2 = st.columns(2)
    with col1:
        thickness = st.number_input(f"Layer {i+1} Thickness (in)", value=0.2, key=f"t_{i}")
    with col2:
        impedance = st.number_input(f"Layer {i+1} Impedance (MRayl)", value=2.5, key=f"z_{i}")
    layer_data.append((thickness, impedance))

# Defect
st.header("Defect Simulation")
defect_type = st.selectbox("Defect Type", ["None", "Delamination", "Crack"])
defect_layer = st.slider("Layer with Defect", min_value=1, max_value=num_layers, value=2)

# Simulation function
def simulate_response(layer_data, fluid_impedance, defect_type=None, defect_layer_idx=None):
    v_nominal = 2000  # m/s
    time = []
    amp = []
    depth = 0.00254  # 0.1 inch fluid gap in meters

    time.append(2 * depth / 1480 * 1e6)  # µs
    amp.append(1.0)

    for i, (t_inch, z) in enumerate(layer_data):
        thickness_m = t_inch * 0.0254
        depth += thickness_m
        t = 2 * depth / v_nominal * 1e6  # µs
        a = 1.0

        if defect_type == "Delamination" and i == defect_layer_idx:
            t += 0.6
            a *= 0.6
        elif defect_type == "Crack" and i == defect_layer_idx:
            a *= 0.5

        time.append(t)
        amp.append(a)

    # Generate A-scan
    time_axis = np.linspace(0, max(time) + 2, 3000)
    signal = np.zeros_like(time_axis)
    for t, a in zip(time, amp):
        signal += a * np.exp(-((time_axis - t) ** 2) / (2 * (0.05 ** 2)))
    signal += 0.02 * np.random.randn(len(signal))  # Noise

    return time_axis, signal

# Run simulation
if st.button("Simulate"):
    time_axis, signal = simulate_response(
        layer_data,
        fluid_impedance,
        defect_type=defect_type if defect_type != "None" else None,
        defect_layer_idx=defect_layer - 1
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, signal, label="A-scan Signal")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Simulated Ultrasonic Response")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
