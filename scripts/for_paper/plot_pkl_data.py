#!/usr/bin/env python3

import numpy as np
import pickle as pkl
import seaborn as sns

import matplotlib as mpl

rc_fonts = {
    "text.usetex": True,
    "mathtext.default": "regular",
    "text.latex.preamble": r"\usepackage{bm}",
}
mpl.rcParams.update(rc_fonts)

import matplotlib.pyplot as plt

abv_color = "#D00000"
blw_color = "#FF8500"
dir_color = "#005F73"
thickness = 8
start = 0
end = 235


def load_pickles(fname):
    ret = None
    with open(fname, "rb") as f:
        while True:
            try:
                ret = pkl.load(f)
                break
            except EOFError:
                break
    return ret


def plot_alpha_and_dir(data):
    alpha_abv = []
    alpha_blw = []
    alpha_dot_abv = []
    alpha_dot_blw = []

    for i in range(len(data["/h_abv"])):
        if i < start:
            continue
        if i > end:
            break

        alpha_abv.append(data["/cbf_alpha_abv"][i][1].data)
        alpha_blw.append(data["/cbf_alpha_blw"][i][1].data)

        alpha_dot_abv.append(data["/alpha_dot_abv"][i][1].data)
        alpha_dot_blw.append(data["/alpha_dot_blw"][i][1].data)

    alpha_abv = np.array(alpha_abv)
    alpha_blw = np.array(alpha_blw)
    alpha_dot_abv = np.array(alpha_dot_abv)
    alpha_dot_blw = np.array(alpha_dot_blw)

    # set color palette
    sns.set_style("whitegrid")

    # plot from h_abv and h_blw
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # ax[0].axhline(y=0, linestyle="--", color="grey", linewidth=5, label="Safety Bound")
    # ax[1].axhline(y=0, linestyle="--", color="grey", linewidth=5, label="Safety Bound")

    sns.lineplot(
        x=np.arange(len(alpha_abv)),
        y=alpha_abv,
        ax=ax[0],
        color=abv_color,
        # label=r"$\overline{\alpha}$",
        linewidth=thickness,
    )
    sns.lineplot(
        x=np.arange(len(alpha_dot_abv)),
        y=alpha_dot_abv,
        ax=ax[0],
        color=dir_color,
        # label=r"$\overline{\dot{\alpha}}$",
        linewidth=thickness,
    )

    sns.lineplot(
        x=np.arange(len(alpha_blw)),
        y=alpha_blw,
        ax=ax[1],
        color=blw_color,
        # label=r"$\underline{\alpha}$",
        linewidth=thickness,
    )
    sns.lineplot(
        x=np.arange(len(alpha_dot_blw)),
        y=alpha_dot_blw,
        ax=ax[1],
        color=dir_color,
        # label=r"$\underline{\dot{\alpha}}$",
        linewidth=thickness,
    )

    # ax[0].set_ylim(-0.5, 2.5)
    ax[1].set_ylim(-0.5, 2.5)

    # ax[0].set_title("CBF Parameter Adaptation")
    # ax[0].legend()


def plot_h_values(data):
    h_abv = []
    h_blw = []
    for i in range(len(data["/h_abv"])):
        if i < start:
            continue
        if i > end:
            break

        h_abv.append(data["/h_abv"][i][1].data)
        h_blw.append(data["/h_blw"][i][1].data)

    h_abv = np.array(h_abv)
    h_blw = np.array(h_blw)

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.axhline(
        y=0, linestyle="--", color="grey", linewidth=7.5
    )  # , label="Safety Bound")
    # ax[1].axhline(
    #     y=0, linestyle="--", color="grey", linewidth=7.5
    # )  # , label="Safety Bound")

    sns.lineplot(
        x=np.arange(len(h_abv)),
        y=h_abv,
        ax=ax,
        color=abv_color,
        # label=r"$\overline{h}$",
        linewidth=thickness,
    )

    sns.lineplot(
        x=np.arange(len(h_abv)),
        y=h_blw,
        ax=ax,
        color=blw_color,
        # label=r"$\underline{h}$",
        linewidth=thickness,
    )

    # set x and y limits
    # ax.set_ylim(-0.02, 0.5)
    # ax[1].set_ylim(-0.02, 0.5)


def main():
    # data = load_pickles("/home/bezzo/catkin_ws/src/mpcc/bags/perfect_sac_run_105.pkl")
    # data = load_pickles(
    #     "/home/nick/catkin_ws/src/mpcc/bags/great_world_294_run_sac.pkl"
    # )
    data = load_pickles("/home/nick/Downloads/take2_results.pkl")

    plot_alpha_and_dir(data)
    plt.tight_layout()

    plot_h_values(data)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
