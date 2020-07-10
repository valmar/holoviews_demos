#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import collections
import itertools
import threading
from typing import Any, Deque, Dict, List

import holoviews  # type: ignore
import holoviews.plotting.bokeh  # type: ignore
import msgpack  # type: ignore
import pandas  # type: ignore
import zmq  # type: ignore
from bokeh.io import curdoc  # type: ignore
from holoviews.operation.datashader import datashade, spread  # type: ignore
from tornado import gen
from tornado.ioloop import PeriodicCallback

renderer = holoviews.renderer("bokeh")


# In[ ]:


class ZMQListener(threading.Thread):
    """See init method."""

    def __init__(self, host, port):
        # type: (str, int) -> None
        """Thread reading data from ZMQ."""
        threading.Thread.__init__(self)
        self.setDaemon(True)

        self.new_data_queue = collections.deque(
            maxlen=1
        )  # type: Deque[List[Dict[bytes, Any]]]

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect("tcp://{0}:{1}".format(host, port))
        self._socket.set_hwm(1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"omdata")
        print("Connected to tcp://{0}:{1}".format(host, port))

        self.start()

    def run(self):
        # type: () -> None
        """Receive and decode data from zmq socket."""
        while True:
            full_message = self._socket.recv_multipart()
            msgpack_message = full_message[1]
            message = msgpack.unpackb(msgpack_message)
            self.new_data_queue.append(message)


# In[ ]:


listening_thread = ZMQListener("127.0.0.1", 8999)


# In[ ]:


hitratestream = holoviews.streams.Buffer(
    pandas.DataFrame({"hitrate": [0.0] * 5000}),
    length=5000,
    following=False,
    index=False,
)


# In[ ]:


peakstream = holoviews.streams.Buffer(
    pandas.DataFrame(
        {"ss": [-1.0, 0.0, 1.0] + [0] * 499997, "fs": [-1.0, 0.0, 1.0] + [0] * 499997}
    ),
    length=500000,
    following=False,
    index=False,
)


# In[ ]:


def create_curve(data):  # type: ignore
    """
    Create Curve for HoloView's Dynamic Map.

    Replaces x axis with event range
    """
    return holoviews.Curve((range(-5000, 0), data.hitrate))


# In[ ]:


hitrate_dmap = holoviews.DynamicMap(create_curve, streams=[hitratestream]).opts(
    xlim=(-5000, 0), ylim=(0, 100), width=500, title="Hitrate History"
)


# In[ ]:


peak_dmap = spread(
    datashade(holoviews.DynamicMap(holoviews.Points, streams=[peakstream]),), px=2,
).opts(
    xlim=(-3000, 3000),
    ylim=(-3000, 3000),
    xaxis="top",
    invert_yaxis=True,
    height=500,
    width=500,
    title="Detected Peaks",
)


# In[ ]:


@gen.coroutine
def update_hitrate_stream():
    # type: () -> None
    print("Receiving")
    try:
        new_data = (
            listening_thread.new_data_queue.popleft()
        )  # type: List[Dict[bytes, Any]]
        print("Got")
        print(pandas.DataFrame({"hitrate": [entry[b"hit_rate"] for entry in new_data]}))
        hitratestream.send(
            pandas.DataFrame({"hitrate": [entry[b"hit_rate"] for entry in new_data]})
        )
        print("Got2")
        peakstream.send(
            pandas.DataFrame(
                {
                    "ss": list(
                        itertools.chain(
                            *(entry[b"peak_list"][b"ss"] for entry in new_data)
                        )
                    ),
                    "fs": list(
                        itertools.chain(
                            *(entry[b"peak_list"][b"fs"] for entry in new_data)
                        )
                    ),
                }
            )
        )
        print("Got3")
    except IndexError:
        pass


# In[ ]:


cb = PeriodicCallback(update_hitrate_stream, 2000)
cb.start()


def reset_plots():
    # type: () -> None
    """Reset all plots to initial state."""
    hitratestream.send(pandas.DataFrame({"hitrate": [0.0] * 5000}))
    peakstream.send(
        pandas.DataFrame(
            {
                "ss": [-1.0, 0.0, 1.0] + [0] * 499997,
                "fs": [-1.0, 0.0, 1.0] + [0] * 499997,
            }
        ),
    )


# In[ ]:


button = bokeh.models.Button(label="Reset Plots", width=100)
button.on_click(reset_plots)


# In[ ]:

layout = peak_dmap + hitrate_dmap

# hitrate_dmap_plot = renderer.get_plot(hitrate_dmap)
# peak_dmap_plot = renderer.get_plot(peak_dmap)
# layout = bokeh.layouts.row(
#     peak_dmap_plot.state,
#     bokeh.layouts.column(
#         hitrate_dmap_plot.state,
#         bokeh.layouts.row(button, align="center"),
#         align="center",
#     ),
# )


# In[ ]:


curdoc().add_root(layout)
