"""This module contains OM's web interface."""
from __future__ import absolute_import, division, print_function

import collections
import threading
from typing import Any, Deque, Dict, List, Tuple

import dash  # type: ignore
import dash.dependencies as dash_deps  # type: ignore
import dash_core_components as dash_core  # type: ignore
import dash_html_components as dash_html  # type: ignore
import datashader  # type: ignore
import h5py  # type: ignore
import msgpack  # type: ignore
import numpy  # type: ignore
import pandas  # type: ignore
import zmq  # type: ignore

# NEEDED FOR DEBUGGING
# from PIL import Image  # type: ignore
# import datashader.reductions as dshader_reduct  # type: ignore
# import random

# import xarray

MAXLEN = 5000  # type: int


class ZMQListener(threading.Thread):
    """See init method."""

    def __init__(self, host, port):
        # type: (str, int) -> None
        """Thread reading data from ZMQ."""
        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.new_data_queue = collections.deque(
            maxlen=1
        )  # type: Deque[List[Dict[bytes, int]]]

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect("tcp://{0}:{1}".format(host, port))
        self._socket.set_hwm(1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"omdata")
        print("Connected to tcp://{0}:{1}".format(host, port))

        self.start()

    def run(self):
        # type: () -> None
        """Function that just receives data."""
        while True:
            full_message = self._socket.recv_multipart()
            msgpack_message = full_message[1]
            message = msgpack.unpackb(msgpack_message)
            self.new_data_queue.append(message)


# def generate_heatmap_image(data, x_range, y_range, plot_width, plot_height):
#     # type: (numpy.ndarray, Tuple[int, int], Tuple[int, int], int, int) -> numpy.ndarray
#     """Function that computes the downsampled map."""
#     if x_range is None or y_range is None or plot_width is None or plot_height is None:
#         return None
#     xarray_data = xarray.DataArray(data, dims=("y", "x"))
#     canvas = datashader.Canvas(
#         x_range=x_range,
#         y_range=y_range,
#         plot_width=plot_width,
#         plot_height=plot_height,
#     )
#     agg_scatter = canvas.raster(xarray_data, downsample_method=dshader_reduct.max())
#     return numpy.array(agg_scatter)


def generate_heatmap_image(
    peak_x,  # type: Deque[int]
    peak_y,  # type: Deque[int]
    x_range,  # type: Tuple[int, int]
    y_range,  # type: Tuple[int, int]
    plot_width,  # type: int
    plot_height,  # type: int
):
    # type: (...) -> numpy.ndarray
    """Function that computes the downsampled map."""
    if x_range is None or y_range is None or plot_width is None or plot_height is None:
        return None
    peak_data = pandas.DataFrame({"x": peak_x, "y": peak_y})
    canvas = datashader.Canvas(
        x_range=x_range,
        y_range=y_range,
        plot_width=plot_width,
        plot_height=plot_height,
    )
    agg_scatter = canvas.points(peak_data, x="x", y="y")
    return numpy.array(agg_scatter)


geometry_filename = "pixelmaps.h5"
with h5py.File(geometry_filename, "r") as fh:
    x_pixelmap = fh["/x_pixelmap"][:]  # type: numpy.ndarray
    y_pixelmap = fh["/y_pixelmap"][:]  # type: numpy.ndarray

max_pixel_x = numpy.max(x_pixelmap)
max_pixel_y = numpy.max(y_pixelmap)

listening_thread = ZMQListener("127.0.0.1", 8999)  # type: ZMQListener

peak_x = collections.deque([])  # type: Deque[int]
peak_y = collections.deque([])  # type: Deque[int]

# DATA FOR DEBUGGING
# sample_image = Image.open("wallpapersden.com_moon-from-space-4k_6000x6000.jpg")
# peak_image = numpy.array(sample_image.convert("L"))
# random_peak_x = tuple(
#     random.randint(0, 5999) for x in range(0, 350000)
# )  # type: Tuple[int, ...]
# random_peak_y = tuple(
#     random.randint(0, 5999) for x in range(0, 350000)
# )  # type: Tuple[int, ...]
#
# peak_x = collections.deque(random_peak_x, maxlen=350000)
# peak_y = collections.deque(random_peak_x, maxlen=350000)


# heatmap_image = generate_heatmap_image(
#     data=peak_image,
#     x_range=(0, 6000),
#     y_range=(0, 6000),
#     plot_width=800,
#     plot_height=800,
# )

heatmap_image = generate_heatmap_image(
    peak_x=peak_x,
    peak_y=peak_y,
    x_range=(0, 6000),
    y_range=(0, 6000),
    plot_width=800,
    plot_height=800,
)

app = dash.Dash(__name__)
server = app.server
app.layout = dash_html.Div(
    dash_html.Div(
        [
            dash_html.H3(children="OM Crystallography GUI"),
            dash_core.Graph(
                id="hit-rate-history-graph",
                figure={
                    "data": [
                        {
                            "x": tuple(range(0, MAXLEN)),
                            "y": [0] * MAXLEN,
                            "type": "scatter",
                        }
                    ],
                },
            ),
            dash_html.Div(
                dash_html.Button("Reset Plot", id="reset-plot-button", n_clicks=0),
            ),
            dash_core.Graph(
                id="virtual-powder-pattern-heatmap",
                figure={
                    "data": [
                        {
                            "z": heatmap_image,
                            "type": "heatmap",
                            "colorscale": "Greys",
                            "x0": 0,
                            "dx": 6000.0 / 800.0,
                            "y0": 0,
                            "dy": 6000.0 / 800.0,
                        }
                    ],
                    "layout": {
                        "width": 800,
                        "height": 800,
                        "yaxis": {"autorange": "reversed", "scaleanchor": "x"},
                    },
                },
            ),
            dash_html.Div(0, id="new_x0", style={"display": "none"}),
            dash_html.Div(800, id="new_x1", style={"display": "none"}),
            dash_html.Div(0, id="new_y0", style={"display": "none"}),
            dash_html.Div(800, id="new_y1", style={"display": "none"}),
            dash_core.Interval(
                id="interval-component",
                interval=1 * 500,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )
)


@app.callback(  # type: ignore
    [
        dash_deps.Output("hit-rate-history-graph", "extendData"),
        dash_deps.Output("virtual-powder-pattern-heatmap", "figure"),
    ],
    [dash_deps.Input("interval-component", "n_intervals")],
    [
        dash_deps.State("hit-rate-history-graph", "figure"),
        dash_deps.State("new_x0", "children"),
        dash_deps.State("new_x1", "children"),
        dash_deps.State("new_y0", "children"),
        dash_deps.State("new_y1", "children"),
    ],
)
def update_hit_plots(
    _,  # type: Any
    current_hit_rate_plot_figure,  # type: Any
    new_x0,  # type: int
    new_x1,  # type: int
    new_y0,  # type: int
    new_y1,  # type: int
):
    # type: (...) -> Tuple[Any, Any]
    """Function to update or reset the hit rate plot."""
    try:
        new_data = (
            listening_thread.new_data_queue.popleft()
        )  # type: List[Dict[bytes, int]]
        for entry in new_data:
            new_x = current_hit_rate_plot_figure["data"][0]["x"][-1] + 1
            new_y = entry[b"hit_rate"]
        hit_rate_ret_val = (
            {"x": [[new_x]], "y": [[new_y]]},
            0,
            MAXLEN,
        )  # type: Tuple[Dict[str,Any], int, int]
    except IndexError:
        hit_rate_ret_val = dash.no_update

    if (new_x1 - new_x0) > 800:
        heatmap_image = generate_heatmap_image(
            peak_x=peak_x,
            peak_y=peak_y,
            x_range=(new_x0, new_x1),
            y_range=(new_y1, new_y0),
            plot_width=800,
            plot_height=800,
        )

        virtual_pow_pattern_ret_val = {
            "data": [
                {
                    "z": heatmap_image,
                    "type": "heatmap",
                    "colorscale": "Greys",
                    "x0": new_x0,
                    "dx": (new_x1 - new_x0) / 800.0,
                    "y0": new_y1,
                    "dy": (new_y0 - new_y1) / 800.0,
                    "zauto": False,
                }
            ],
            "layout": {
                "width": 800,
                "height": 800,
                "yaxis": {"autorange": "reversed", "scaleanchor": "x"},
            },
        }
    else:
        virtual_pow_pattern_ret_val = dash.no_update

    return hit_rate_ret_val, virtual_pow_pattern_ret_val


@app.callback(  # type: ignore
    dash_deps.Output("hit-rate-history-graph", "figure"),
    [dash_deps.Input("reset-plot-button", "n_clicks")],
)
def reset_hit_rate_history(_):
    # type: (Any) -> Any
    """Function to update or reset the hit rate plot."""
    return {
        "data": [{"x": tuple(range(0, MAXLEN)), "y": [0] * MAXLEN, "type": "scatter"}],
    }


@app.callback(  # type: ignore
    [
        dash_deps.Output("new_x0", "children"),
        dash_deps.Output("new_x1", "children"),
        dash_deps.Output("new_y0", "children"),
        dash_deps.Output("new_y1", "children"),
    ],
    [dash_deps.Input("virtual-powder-pattern-heatmap", "relayoutData")],
    [dash_deps.State("virtual-powder-pattern-heatmap", "figure")],
)
def selectionHighlight(relayout_data, figure):
    # type: (Dict[str, Any], Dict[str, Any]) -> Any
    """Function that reacts to panning/zooming."""
    if relayout_data is not None and figure.keys():
        if "xaxis.range[0]" in relayout_data:
            new_x0 = int(round(relayout_data["xaxis.range[0]"]))
        else:
            new_x0 = int(round(figure["data"][0]["x0"]))
        if "xaxis.range[1]" in relayout_data:
            new_x1 = int(round(relayout_data["xaxis.range[1]"]))
        else:
            new_x1 = int(round(figure["data"][0]["x0"] + figure["data"][0]["dx"] * 800))
        if "yaxis.range[0]" in relayout_data:
            new_y0 = int(round(relayout_data["yaxis.range[0]"]))
        else:
            new_y0 = int(round(figure["data"][0]["y0"]))
        if "yaxis.range[1]" in relayout_data:
            new_y1 = int(round(relayout_data["yaxis.range[1]"]))
        else:
            new_y1 = int(round(figure["data"][0]["y0"] + figure["data"][0]["dy"] * 800))

        return new_x0, new_x1, new_y0, new_y1
    else:
        return dash.no_update


if __name__ == "__main__":

    app.run_server(debug=True)
