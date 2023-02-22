/* GStreamer
 * Copyright (C) 2023 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_FDTRAKER_H_
#define _GST_FDTRAKER_H_

#include <gst/base/gstbasetransform.h>
#include <queue>
#include <vector>
#include "gstnvdsmeta.h"

G_BEGIN_DECLS

#define GST_TYPE_FDTRACKER   (gst_fdtracker_get_type())
#define GST_FDTRACKER(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_FDTRACKER,GstFdtracker))
#define GST_FDTRAKERCLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_FDTRAKERGstFdtrackerClass))
#define GST_IS_FDTRACKER(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_FDTRACKER))
#define GST_IS_FDTRACKER_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_FDTRACKER))

typedef struct _GstFdtracker GstFdtracker;
typedef struct _GstFdtrackerClass GstFdtrackerClass;

struct Bbox_cache {
  float x1;
  float y1;
  float x2;
  float y2;
  int tracker_id = -1;
};

struct _GstFdtracker
{
  GstBaseTransform base_fdtracker;

  std::queue<std::vector<Bbox_cache>>* previous_frame;
  int min_tracker_id = 0;
};

struct _GstFdtrackerClass
{
  GstBaseTransformClass base_fdtracker_class;
};

GType gst_fdtracker_get_type (void);

G_END_DECLS

#endif