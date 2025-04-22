import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import List

ALPHA_VALUE = 0.75
PARAM_LIMS = [-1, 1]
OBSERVABLES = ["X", "Y", "Z"]
PARENT_DIR = "../feature_identification_paper/figures/"

SMALL_FONTSIZE = 18
MEDIUM_FONTSIZE = SMALL_FONTSIZE + 2
LARGE_FONTSIZE = MEDIUM_FONTSIZE + 2

plt.rc("font", family="serif", serif="cm10")
plt.rc("text", usetex=True)


def plotting_wrapper_func(
    title: str,
    data_frame: np.ndarray,
    labels: List[str],
    label_title: str,
    file_name: str,
    use_pdf: bool = False,
):
    if use_pdf:
        mpl.use("pdf")

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title, fontsize=LARGE_FONTSIZE)

    for observable_index, observable in enumerate(OBSERVABLES):
        ax = fig.add_subplot(1, 3, observable_index + 1, projection="3d")
        ax.clear()
        ax.set_title(f"$Q_{observable}$", fontsize=MEDIUM_FONTSIZE)

        ax.set_xlim3d(*PARAM_LIMS)
        ax.set_ylim3d(*PARAM_LIMS)
        ax.set_zlim3d(*PARAM_LIMS)
        ax.set_xlabel(r"$\alpha$", fontsize=SMALL_FONTSIZE)
        ax.set_ylabel(r"$\beta$", fontsize=SMALL_FONTSIZE)
        ax.set_zlabel(r"$\gamma$", fontsize=SMALL_FONTSIZE)
        ax.set_xticks([PARAM_LIMS[0], 0, PARAM_LIMS[1]])
        ax.set_yticks([PARAM_LIMS[0], 0, PARAM_LIMS[1]])
        ax.set_zticks([PARAM_LIMS[0], 0, PARAM_LIMS[1]])
        ax.tick_params(labelsize=16)

        base_idx = observable_index * 3

        for profile in data_frame:
            ax.scatter(
                profile[base_idx],
                profile[base_idx + 1],
                profile[base_idx + 2],
                alpha=ALPHA_VALUE,
                s=100,
            )

    fig.legend(
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.525, -0.125),
        title=label_title,
        title_fontsize=MEDIUM_FONTSIZE,
        prop={"size": SMALL_FONTSIZE},
    )
    fig.subplots_adjust(hspace=0.4)

    if use_pdf:
        plt.savefig(f"{PARENT_DIR}{file_name}.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()
        plt.close()
