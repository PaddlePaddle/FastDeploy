// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STREAMER_SRC_GSTREAMER_PLUGIN_FDINFER_FDINFER_H_
#define STREAMER_SRC_GSTREAMER_PLUGIN_FDINFER_FDINFER_H_

#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

#define GST_TYPE_FDINFER (gst_fdinfer_get_type())
#define GST_FDINFER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FDINFER, GstFdinfer))
#define GST_FDINFER_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_FDINFER, GstFdinferClass))
#define GST_IS_FDINFER(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_FDINFER))
#define GST_IS_FDINFER_CLASS(obj) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_FDINFER))

typedef struct _GstFdinfer GstFdinfer;
typedef struct _GstFdinferClass GstFdinferClass;

struct _GstFdinfer {
  GstBaseTransform base_fdinfer;
  void* model;
  gchar* model_name;
  gchar* model_dir;
};

struct _GstFdinferClass {
  GstBaseTransformClass base_fdinfer_class;
};

GType gst_fdinfer_get_type(void);

G_END_DECLS

#endif  // STREAMER_SRC_GSTREAMER_PLUGIN_FDINFER_FDINFER_H_
