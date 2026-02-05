#include "kvcache_connection.h"
#include "kvcache_rdma.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(rdma_comm, m) {
  m.doc() = R"pbdoc(kv cache messager)pbdoc";
  py::class_<RDMACommunicator>(m, "RDMACommunicator")
      .def(py::init<std::string &,
                    int,
                    std::string &,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    int,
                    int,
                    std::vector<int64_t>,
                    std::vector<int64_t>,
                    int,
                    int,
                    int>(),
           py::arg("splitwise_role"),
           py::arg("gpu_idx"),
           py::arg("port"),
           py::arg("key_cache_ptrs"),
           py::arg("value_cache_ptrs"),
           py::arg("block_number"),
           py::arg("block_bytes"),
           py::arg("key_scale_ptrs") = std::vector<int64_t>{},
           py::arg("value_scale_ptrs") = std::vector<int64_t>{},
           py::arg("scale_block_bytes") = 0,
           py::arg("prefill_tp_size") = 1,
           py::arg("prefill_tp_idx") = 0)
      .def("connect",
           &RDMACommunicator::connect,
           py::arg("dst_ip"),
           py::arg("dst_port"),
           py::arg("dst_tp_size") =
               0,  // Default 0: assumes dest has same tp_size as source;
                   // otherwise specifies decode tp_size
           py::call_guard<py::gil_scoped_release>())
      .def("is_connected",
           &RDMACommunicator::is_connected,
           py::arg("dst_ip"),
           py::arg("dst_port"),
           py::call_guard<py::gil_scoped_release>())
      .def("write_cache",
           &RDMACommunicator::write_cache,
           py::arg("dst_ip"),
           py::arg("dst_port"),
           py::arg("local_block_ids"),
           py::arg("remote_block_ids"),
           py::arg("layer_idx"),
           py::call_guard<py::gil_scoped_release>());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
