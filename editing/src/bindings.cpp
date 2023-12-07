#include <torch/extension.h>

#include "editgrid.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("edit_grid", &edit_grid, "load_edit_grid (C++)");
    m.def("load_float_vec", &load_float_vec, "load_edit_grid_pooints (C++)");
}