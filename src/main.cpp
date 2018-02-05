#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "MIQP.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyMIQP, m)
{
  m.doc() = "MIQP";

  py::class_<MIQP>(m, "MIQP")
  .def(py::init<>())
  .def("set_c", &MIQP::set_c, "set_c")
  .def("set_Q", &MIQP::set_Q, "set_Q")
  .def("set_A", &MIQP::set_A, "set_A")
  .def("set_glb", &MIQP::set_glb, "set_glb")
  .def("set_gub", &MIQP::set_gub, "set_gub")
  .def("set_xlb", &MIQP::set_xlb, "set_xlb")
  .def("set_xub", &MIQP::set_xub, "set_xub")
  .def("set_var_types", &MIQP::set_var_types, "set_var_types")
  .def("solve_bb", &MIQP::solve_bb, "solve_bb")
  .def("set_initial_point", &MIQP::set_initial_point, "set_initial_point")
  .def("get_sol_x", &MIQP::get_sol_x, "get_sol_x")
  .def("get_sol_obj", &MIQP::get_sol_obj, "get_sol_obj")
  .def("get_sol_status", &MIQP::get_sol_status, "get_sol_status")
  .def("get_sol_time", &MIQP::get_sol_time, "get_sol_time")
  ;
}
