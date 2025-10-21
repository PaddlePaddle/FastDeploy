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
                    int>())
      .def("connect", &RDMACommunicator::connect)
      .def("is_connected", &RDMACommunicator::is_connected)
      .def("write_cache", &RDMACommunicator::write_cache);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
