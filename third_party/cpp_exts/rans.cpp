#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_ops(py::module_ &m);
void init_buffered_coders(py::module_ &m);
void init_encoders(py::module_ &m);
void init_decoders(py::module_ &m);


PYBIND11_MODULE(rans, m) {
  m.attr("__name__") = "mcquic.rans";

  m.doc() = "range Asymmetric Numeral System python bindings";

  init_ops(m);
  init_buffered_coders(m);
  init_encoders(m);
  init_decoders(m);
}
