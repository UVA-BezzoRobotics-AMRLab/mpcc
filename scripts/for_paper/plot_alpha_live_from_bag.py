#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import cm
import rospy
from std_msgs.msg import Float64
from matplotlib.lines import Line2D

import matplotlib as mpl

rc_fonts = {
    "text.usetex": True,
    "mathtext.default": "regular",
    "text.latex.preamble": r"\usepackage{bm}",
}
mpl.rcParams.update(rc_fonts)

# Initialize global variables for plot updates
timestamps = []
values = []
fig, ax = plt.subplots(figsize=(10, 6))
# norm = plt.Normalize(0, 1)  # Initial normalization, will update
# lc = LineCollection([], cmap='cool', norm=norm, linewidth=5)
# line = ax.add_collection(lc)
lc = None
norm = None

initial_time = 0
final_time = 23.5

start = None


# def normalize_timestamps(timestamps):
#     global initial_time, final_time
#     return np.array([(t - initial_time) / (final_time - initial_time) for t in timestamps])


def update_plot(data):
    global timestamps, values, lc, norm, start
    # Append new data
    if start is None:
        start = rospy.Time.now().to_sec()

    timestamps.append(rospy.Time.now().to_sec() - start)
    values.append(data.data)

    # Update line segments
    points = np.array([timestamps, values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)

    # Update normalization and colormap
    lc.set_array(timestamps)

    ax.autoscale()

    plt.draw()


def listener(topic_name):
    global fig, ax, lc, norm
    # Initialize ROS node
    rospy.init_node("plot_alpha_live", anonymous=True)

    # Setup plot
    ax.set_title(
        r"$\bm{\underline{\alpha}}$ \textbf{vs Time}", fontsize=20, fontweight="bold"
    )

    norm = plt.Normalize(initial_time, final_time)
    lc = LineCollection([], colors="#FF8500", norm=norm, linewidth=5)
    line = ax.add_collection(lc)

    # Adjust plot limits
    ax.set_xlim(0, final_time)
    ax.set_ylim(0, 2.5)

    ax.set_xlabel(r"\textbf{Time (s)}", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"{${\bm{\underline{\alpha}}}$}", fontsize=18)
    ax.set_title(
        r"$\bm{\underline{\alpha}}$ \textbf{vs Time}", fontsize=20, fontweight="bold"
    )

    ax.tick_params(axis="both", which="major", labelsize=20)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    cmap = plt.get_cmap("cool")
    legend_color = cmap(0.5)

    legend_line = Line2D(
        [0], [0], color="#FF8500", linewidth=5, label=r"$\bm{\underline{\alpha}(t)}$"
    )

    ax.legend(
        handles=[legend_line], prop={"weight": "bold", "size": 20}, loc="upper right"
    )

    # Subscribe to the topic
    rospy.Subscriber(topic_name, Float64, update_plot)

    plt.ion()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    topic_name = "/cbf_alpha_blw"
    listener(topic_name)
