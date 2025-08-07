import plotly.graph_objs as go

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


