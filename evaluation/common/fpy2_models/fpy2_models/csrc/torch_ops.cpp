#include <Python.h>

// The operators are generated with the models; see generate_models.py.

extern "C" PyObject* PyInit__C(void) {
  static PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr,
  };
  return PyModule_Create(&module);
}
