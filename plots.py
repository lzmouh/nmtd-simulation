import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc
import numpy as np

def draw_nmted_schematics(layer_data, Z_fluid,
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
                 f"{label}\nZ={Z:.2f}\n{t:.2f}\"",
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

# Example usage
layers = [
    ("Layer 1", 0.2, 2.5),
    ("Layer 2", 0.3, 3.0),
    ("Layer 3", 0.25, 3.4),
    ("Layer 4", 0.25, 3.6),
    ("Layer 5", 0.2, 4.0),
]
fig = draw_nmted_schematics(layers, Z_fluid=1.48,
                            defect_type="Delamination", defect_layer=2)
plt.show()
