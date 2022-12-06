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

#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {
std::string GetElementName(GstElement* elem) {
  gchar* name = gst_element_get_name(elem);
  std::string res(name);
  g_free(name);
  return res;
}

std::vector<std::string> GetSinkElemNames(GstBin* bin) {
  GstIterator *it;
  GValue val = G_VALUE_INIT;
  gboolean done = FALSE;
  std::vector<std::string> names;

  it = gst_bin_iterate_sinks(bin);
  do {
    switch (gst_iterator_next(it, &val)) {
      case GST_ITERATOR_OK: {
        GstElement* sink = static_cast<GstElement*>(g_value_get_object(&val));
        names.push_back(GetElementName(sink));
        g_value_reset(&val);
        break;
      }
      case GST_ITERATOR_RESYNC:
        gst_iterator_resync(it);
        break;
      case GST_ITERATOR_ERROR:
        GST_ERROR("Error iterating over %s's sink elements",
            GST_ELEMENT_NAME(bin));
      case GST_ITERATOR_DONE:
        g_value_unset(&val);
        done = TRUE;
        break;
    }
  } while (!done);

  gst_iterator_free(it);
  return names;
}

}  // namespace streamer
}  // namespace fastdeploy
