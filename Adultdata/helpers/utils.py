import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
)
import seaborn as sns
import matplotlib.pyplot as plt


def plot_fairness_and_accuracy(results, group="sex", title="Accuracy and Fairness vs λ"):
    sns.set(style="whitegrid", font_scale=1.2)
    color1= "#fca483"
    color2 = "#71c5ab"
    color3= "#daa156"

    lambdas = results["lambda"]
    acc = [100 * a for a in results["test_accuracy"]]
    dp = results[f"dp_diff_{group}"]
    eo = results[f"eo_diff_{group}"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # First Y-axis: Accuracy
    ax1.plot(lambdas, acc, marker="o", label="Accuracy", color=color2)
    ax1.set_ylabel("Test Accuracy (%)", color=color2)
    ax1.set_xlabel(r"$\lambda$ (Fairness penalty)")

    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor=color2)

    # Second Y-axis: Fairness
    ax2 = ax1.twinx()
    ax2.plot(lambdas, dp, marker="s", label="DP difference", color=color1)
    ax2.plot(lambdas, eo, marker="^", label="EO difference", color=color3)


    ax2.set_ylabel(r"Fairness violation ($\downarrow$ better)", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    ax2.axhline(0.05, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylim(0, 0.3)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def bin_hours_per_week(hpw):
    """
    Bin the hours per week variable into 0-30, 31-40, 41-50, 50+ bins.
    """
    if hpw <= 30:
        return 0
    elif hpw <= 40:
        return 1
    elif hpw <= 50:
        return 2
    return 3




COLORS = ["#f2603b", "#262445", "#00a886", "#edc946", "#70cfcf"]
LINE_STYLES = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
TRANSPARENT = "rgba(0,0,0,0)"
GRID_COLOR = "rgb(159, 197, 232)"


def _hex_to_rgba(hex_code, a=0.3):
    def cast(s):
        return int(s, 16)

    r = cast(hex_code[1:3])
    g = cast(hex_code[3:5])
    b = cast(hex_code[5:7])
    return f"rgba({r},{g},{b},{a})"


def group_box_plots(
    scores,
    attr,
    groups=None,
    group_names=None,
    title="",
    xlabel="",
    ylabel="",
):
    """
    Helper function for plotting group box curves. Assumes binary labels.
    """
    if groups is None:
        groups = np.zeros_like(scores)
        group_names = [""]

    unique_groups = sorted(set(groups))

    # Outline colours predefined for adjustability
    x_grid_color = GRID_COLOR
    y_grid_color = TRANSPARENT
    x_zero_line_color = x_grid_color
    y_zero_line_color = TRANSPARENT
    # Background colours
    paper_bgcolor = TRANSPARENT
    plot_bgcolor = TRANSPARENT

    return go.Figure(
        data=[
            go.Box(
                x=scores[attr == a],
                y=groups[attr == a],
                name=a,
                orientation="h",
                marker={"color": _hex_to_rgba(COLORS[i], 1), "opacity": 0.1},
                line_color=COLORS[i],
                hoverinfo="name+x",
                jitter=0.2,
            )
            for i, a in enumerate(sorted(set(attr)))
        ],
        layout={
            "autosize": True,
            "boxmode": "group",
            "height": 200 + 40 * len(set(attr)) * len(set(groups)),
            "hovermode": "closest",
            "title": title,
            "xaxis": {
                "hoverformat": ".3f",
                "title": xlabel,
                "gridcolor": x_grid_color,
                "zerolinecolor": x_zero_line_color,
                "fixedrange": True,
            },
            "yaxis": {
                "tickvals": unique_groups,
                "ticktext": group_names or unique_groups,
                "title": ylabel,
                "gridcolor": y_grid_color,
                "zerolinecolor": y_zero_line_color,
                "fixedrange": True,
            },
            "paper_bgcolor": paper_bgcolor,
            "plot_bgcolor": plot_bgcolor,
        },
    )


    scores,
    attr,
    groups=None,
    group_names=None,
    title="",
    xlabel="",
    ylabel="",
):
    if groups is None:
        groups = np.zeros_like(scores)
        group_names = [""]

    unique_groups = sorted(set(groups))

    # Outline colours predefined for adjustability
    x_grid_color = GRID_COLOR
    y_grid_color = TRANSPARENT
    x_zero_line_color = x_grid_color
    y_zero_line_color = TRANSPARENT
    # Background colours
    paper_bgcolor = TRANSPARENT
    plot_bgcolor = TRANSPARENT

    return go.Figure(
        data=[
            go.Bar(
                x=[
                    scores[(attr == a) & (groups == group)].mean()
                    for group in unique_groups
                ],
                marker={
                    "color": _hex_to_rgba(COLORS[i], 0.5),
                    "line_color": COLORS[i],
                    "line_width": 1,
                },
                y=unique_groups,
                name=a,
                orientation="h",
                hoverinfo="name+x",
            )
            for i, a in enumerate(sorted(set(attr)))
        ],
        layout={
            "autosize": True,
            "barmode": "group",
            "height": 200 + 40 * len(set(attr)) * len(set(groups)),
            "hovermode": "closest",
            "title": title,
            "xaxis": {
                "hoverformat": ".3f",
                "title": xlabel,
                "range": [0, 1],
                "gridcolor": x_grid_color,
                "zerolinecolor": x_zero_line_color,
                "fixedrange": True,
            },
            "yaxis": {
                "tickvals": unique_groups,
                "ticktext": group_names or unique_groups,
                "title": ylabel,
                "gridcolor": y_grid_color,
                "zerolinecolor": y_zero_line_color,
                "fixedrange": True,
            },
            "paper_bgcolor": paper_bgcolor,
            "plot_bgcolor": plot_bgcolor,
        },
    )


    """
    Helper function for plotting group ROC curves. Assumes binary labels.
    """
    rocs = []
    for a in sorted(set(attr)):
        data = roc_curve(labels[attr == a], scores[attr == a])
        thresh_index = min(np.where(data[2] <= 0.5)[0])
        rocs.append({"name": a, "data": data, "thresh_index": thresh_index})

    # Outline colours predefined for adjustability
    x_grid_color = GRID_COLOR
    y_grid_color = x_grid_color
    x_zero_line_color = x_grid_color
    y_zero_line_color = y_grid_color
    # Background colours
    paper_bgcolor = TRANSPARENT
    plot_bgcolor = TRANSPARENT

    return go.Figure(
        data=[
            go.Scatter(x=roc["data"][0], y=roc["data"][1], name=roc["name"])
            for roc in rocs
        ]
        + [
            go.Scatter(
                x=[roc["data"][0][roc["thresh_index"]]],
                y=[roc["data"][1][roc["thresh_index"]]],
                name=f"{roc['name']} - threshold",
                mode="markers",
                marker={"color": COLORS[i], "size": 15},
            )
            for i, roc in enumerate(rocs)
        ],
        layout={
            "autosize": True,
            "xaxis": {
                "title": "False Positive Rate",
                "gridcolor": x_grid_color,
                "zerolinecolor": x_zero_line_color,
                "fixedrange": True,
            },
            "yaxis": {
                "title": "True Positive Rate",
                "gridcolor": y_grid_color,
                "zerolinecolor": y_zero_line_color,
                "fixedrange": True,
            },
            "paper_bgcolor": paper_bgcolor,
            "plot_bgcolor": plot_bgcolor,
        },
    )


def bar_chart(
    x, y, title="", xlabel="", ylabel="", xticks=None, yrange=[0, 1],
):
    """
    Bar chart with consistent styling as well

    x: x values
    y: y values
    title: plot title (optional)
    xlabel: x axis label (optional)
    ylabel: y axis label (optional)
    xticks: dictionary if using custom labels,
            such that{"tickvals": [...], "ticktext": [...]} (optional)
    yrange: the range of the y axis (optionaL)
    """
    # Outline colours predefined for adjustability
    x_grid_color = TRANSPARENT
    y_grid_color = "#71c5ab"
    x_zero_line_color = x_grid_color
    y_zero_line_color = y_grid_color
    # Background colours
    paper_bgcolor = TRANSPARENT
    plot_bgcolor = TRANSPARENT

    if xticks is None:
        xticks = {}
        xticks["tickvals"] = x
        xticks["ticktext"] = [str(val) for val in x]

    return go.Figure(
        [
            go.Bar(
                x=x,
                y=y,
                marker={
                    "color": _hex_to_rgba("#fca483", 0.5),
                    "line_color": COLORS[0],
                    "line_width": 1,
                },
            )
        ],
        layout={
            "autosize": True,
            "width": 500,
            "hovermode": "closest",
            "title": title,
            "xaxis": {
                "title": xlabel,
                "gridcolor": x_grid_color,
                "zerolinecolor": x_zero_line_color,
                "tickvals": xticks["tickvals"],
                "ticktext": xticks["ticktext"],
                "fixedrange": True,
            },
            "yaxis": {
                "title": ylabel,
                "gridcolor": y_grid_color,
                "range": yrange,
                "zerolinecolor": y_zero_line_color,
                "fixedrange": True,
            },
            "paper_bgcolor": paper_bgcolor,
            "plot_bgcolor": plot_bgcolor,
        },
    )





def accuracy(labels, scores, threshold=0.5):
    """
    Computes accuracy from scores and labels. Assumes binary classification.
    """
    return ((scores >= threshold) == labels).mean()


