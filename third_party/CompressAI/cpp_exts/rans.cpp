#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_ops(py::module_ &m);
void init_encoders(py::module_ &m);
void init_decoders(py::module_ &m);


PYBIND11_MODULE(rans, m) {
  m.attr("__name__") = "mcquic.rans";

  m.doc() = R"(range Asymmetric Numeral System (rANS) python bindings.

Exports:
    RansEncoder: Encode list of symbols to string.
    RansDecoder: Decode a string to a list of symbols.
    pmfToQuantizedCDF: Return quantized CDF for a given PMF.)";

  init_ops(m);
  init_encoders(m);
  init_decoders(m);
}
