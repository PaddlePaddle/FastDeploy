# FDStreamer YAML Configuration Guidance

## Design

FDStreamer YAML configuration file = AppConfig + [GStreamer PIPELINE-DESCRIPTION String](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c#pipeline-description)

In other words, FDStreamer YAML configuration file can be parsed into an AppConfig structure and a GStreamer PIPELINE-DESCRIPTION String.

`AppConfig` describes the configuration information at the application level, including App type, whether to enable performance measurement, etc. For details, please refer to the definition of the AppConfig structure in [base_app.h](../../src/app/base_app.h).

GStreamer PIPELINE-DESCRIPTION String is a string defined by the GStreamer framework to describe the Pipeline. Developers can use the command-line tool `gst-launch-1.0` provided by GStreamer to test and verify the PIPELINE-DESCRIPTION string.PIPELINE-DESCRIPTION is composed of Element, Property, Link, etc. Developers can use the command-line tool `gst-inspect-1.0` provided by GStreamer to search for Elements, check Element Properties, etc.More information can be found in [GStreamer Pipeline Description](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c#pipeline-description).

Rules:
- A configuration module of the FDStreamer YAML configuration file, called a `Node`, which consists of a Node name and several Node properties, and the number of Node properties can be 0.
- The first Node must be `app`, used to define AppConfig.
- Except for the [special Node](#special-node-and-property), other Node name must be the name of the GStreamer Element, which can be found by `gst-inspect-1.0` tool.
- Except for the [special Node and properties](#special-node-and-property), other Node property names must be the name of GStreamer Element Properties, which can be checked by `gst-inspect-1.0` tool.

Through the above rules, we can cover all the GStreamer Elements. As long as it can be found by the `gst-inspect-1.0` tool, it can be written into the YAML configuration file of FDStreamer. We can flexibly configure Properties, replace Elements and even modify the Pipeline structure.

## Example

In the following YAML configuration, theare are 5 Nodes, i.e app, nvurisrcbin, nvvideoconvert, capsfilter and appsink.

Except for the app Node, the rest are GStreamer Elements, and the Node name and properties can be queried through the `gst-inspect-1.0` tool. The corresponding GStreamer Pipeline is: `nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=0 ! nvvideoconvert gpu-id=0 ! capsfilter caps="video/x-raw,format=(string)BGR" ! appsink sync=true max-buffers=60 drop=false`

```
app:
  type: video_decoder
  enable-perf-measurement: true
  perf-measurement-interval-sec: 5

nvurisrcbin:
  uri: file:///opt/sample_ride_bike.mov
  gpu-id: 0

nvvideoconvert:
  gpu-id: 0

capsfilter:
  caps: video/x-raw,format=(string)BGR

appsink:
  sync: true
  max-buffers: 60
  drop: false
```

The following example has a nvurisrcbin_list Node, which contains 4 nvurisrcbins, and these 4 nvurisrcbins are connected to one nvstreammux. The corresponding GStreamer Pipeline is: `nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_0  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_1  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_2  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_3  nvstreammux name=mux gpu-id=1 batch-size=4`

```
nvurisrcbin_list:
  uri-list:
  - file:///opt/sample_ride_bike.mov
  - file:///opt/sample_ride_bike.mov
  - file:///opt/sample_ride_bike.mov
  - file:///opt/sample_ride_bike.mov
  pad-prefix: mux.sink_
  gpu-id: 0

nvstreammux:
  name: mux
  gpu-id: 0
  batch-size: 4
```

## Special Node and Property

### app

Used to define AppConfig. For specific configuration items, please refer to the definition of the AppConfig structure in [base_app.h](../../src/app/base_app.h).
Property names are named in lowercase and separated by `-`, and property values are named in lowercase and separated by `_`.

### nvurisrcbin_list

Prior knowledge: GStreamer Pad is the interface of Element, sink pad is the input interface, and src pad is the output interface.

Used to define multiple nvurisrcbins, special properties include:
- uri-list: used to define multiple uri, the value is a YAML list
- pad-prefix: used to define the prefix of the sink pad that the src pad of nvurisrcbin will be connected to, such as mux.sink_, when there are 4 uris in the uri-list, the src pads of the 4 nvurisrcbins are respectively connected to mux.sink_0 , mux.sink_1, mux.sink_2 and mux.sink_3, which are the 4 sink pads of the element named `mux`.

### _link_to

`_link_to` is a special property that sets the name of the sink pad that this Element's src pad will be connected to. This property, if present, must be written as the last property.

`_link_to` is used for a single Element, similar to the function of `pad-prefix`, but `pad-prefix` is used for list.
