// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_GSTFDTRACKER_H_
#define STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_GSTFDTRACKER_H_

#include <gst/base/gstbasetransform.h>
#include <queue>
#include <vector>
#include <map>
#include <gstnvdsmeta.h>
#include "include/ocsort.h"
#include "include/trajectory.h"

G_BEGIN_DECLS

#define GST_TYPE_FDTRACKER   (gst_fdtracker_get_type())
#define GST_FDTRACKER(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FDTRACKER, GstFdtracker))
#define GST_FDTRAKERCLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_FDTRAKERGstFdtrackerClass))
#define GST_IS_FDTRACKER(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_FDTRACKER))
#define GST_IS_FDTRACKER_CLASS(obj) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_FDTRACKER))

typedef struct _GstFdtracker GstFdtracker;
typedef struct _GstFdtrackerClass GstFdtrackerClass;

struct _GstFdtracker {
  GstBaseTransform base_fdtracker;

  std::map<int, OcSortTracker*>* tracker_per_class;
  std::map<int, Trajectory*>* trajectory_per_class;
};

struct _GstFdtrackerClass {
  GstBaseTransformClass base_fdtracker_class;
};

GType gst_fdtracker_get_type(void);

G_END_DECLS

#endif  // STREAMER_SRC_GSTREAMER_PLUGIN_FDTRACKER_GSTFDTRACKER_H_
