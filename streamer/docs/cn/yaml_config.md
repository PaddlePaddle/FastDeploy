# FDStreamer YAML配置文件指南

## 设计思想

FDStreamer的YAML配置文件 = AppConfig + [GStreamer PIPELINE-DESCRIPTION String](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c#pipeline-description)

换言之，FDStreamer的YAML配置文件可以被解析为一个AppConfig结构体和一个GStreamer PIPELINE-DESCRIPTION String。

`AppConfig`描述了应用层面的配置信息，包括App类型、是否打开性能测试等。具体可以查看[base_app.h](../../src/app/base_app.h)中的AppConfig结构体定义。

GStreamer PIPELINE-DESCRIPTION String则是由GStreamer框架定义的用于描述Pipeline的字符串，开发者可以使用GStreamer提供的命令行工具`gst-launch-1.0`对PIPELINE-DESCRIPTION字符串进行测试和验证。PIPELINE-DESCRIPTION由Element、Property、Link等元素组成。开发者可以使用GStreamer提供的命令行工具`gst-inspect-1.0`来搜索Element、查看Element的Property信息等。
更多信息可以查看[GStreamer文档](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html?gi-language=c#pipeline-description)。

规则定义：
- FDStreamer YAML配置文件的一个配置模块，称为一个Node，Node由Node名称和若干个Node属性组成，Node属性个数可以为0
- 第一个Node必须为`app`, 用于定义AppConfig
- 除[特殊定义的Node](#特殊定义的node和属性)外，其余Node的名称必须为GStreamer Element的名称，也就是可以用`gst-inspect-1.0`工具查询到的Element名称
- 除[特殊定义的Node](#特殊定义的node和属性)外，其余Node的属性名称必须为GStreamer Element Properties的名称，也就是可以用`gst-inspect-1.0`工具查询到的Element Properties名称

通过以上的规则，实现了GStreamer Element及其Property的全覆盖，只要可以用`gst-inspect-1.0`工具查询到的，就可以写到FDStreamer的YAML配置文件中。可以灵活地配置Property、替换Element甚至修改Pipeline结构。

## 示例解析

如下的YAML配置文件，由5个Node组成，分别是：app、nvurisrcbin、nvvideoconvert、capsfilter和appsink。

除app Node外，其余都是GStreamer Element，Node名称和属性都可以通过`gst-inspect-1.0`工具查询到。对应的GStreamer Pipeline为：`nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=0 ! nvvideoconvert gpu-id=0 ! capsfilter caps="video/x-raw,format=(string)BGR" ! appsink sync=true max-buffers=60 drop=false`

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

下面的例子有一个nvurisrcbin_list Node，其中包括了4个nvurisrcbin，这4个nvurisrcbin同时接入到了一个nvstreammux。对应的GStreamer Pipeline为：
`nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_0  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_1  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_2  nvurisrcbin uri=file:///opt/sample_ride_bike.mov gpu-id=1 ! mux.sink_3  nvstreammux name=mux gpu-id=1 batch-size=4`

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

## 特殊定义的Node和属性

### app

用于定义AppConfig。具体的配置项可以查看[base_app.h](../../src/app/base_app.h)中的AppConfig结构体定义。
属性名称以小写命名，用`-`分隔，属性值以小写命名，用`_`分隔。

### nvurisrcbin_list

先验知识：GStreamer Pad是Element的接口，sink pad表示输入接口，src pad表示输出接口。

用于定义多个nvurisrcbin，特殊的属性包括：
- uri-list: 用于定义多个uri，值类型为YAML list
- pad-prefix: 用于定义nvurisrcbin的src pad将连接到的sink pad的前缀，例如mux.sink_，当uri-list中有4个uri时，则4个nvurisrcbin的src pad分别连接到了mux.sink_0，mux.sink_1，mux.sink_2，mux.sink_3。而这四个pad则是名字为mux的元素的4个sink pad。

### _link_to

`_link_to`是一个特殊的属性，用于设置该Element的src pad将连接到的sink pad的名称。该属性如果存在，必须写为最后一个属性。

`_link_to`用于单个Element，与`pad-prefix`功能类似，而`pad-prefix`用于list。
