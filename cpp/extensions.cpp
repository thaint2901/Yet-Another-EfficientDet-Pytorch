#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "engine.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

vector<at::Tensor> infer(sample_onnx::Engine &engine, at::Tensor data) {
    CHECK_INPUT(data);

    int batch = data.size(0);
    auto out1 = at::zeros({batch, 64, 64, 64}, data.options());
    auto out2 = at::zeros({batch, 64, 32, 32}, data.options());
    auto out3 = at::zeros({batch, 64, 16, 16}, data.options());

    vector<void *> buffers;
    for (auto buffer : {data, out1, out2, out3}) {
        buffers.push_back(buffer.data<float>());
    }

    engine.infer(buffers, batch);

    return {out1, out2, out3};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<sample_onnx::Engine>(m, "Engine")
        .def(pybind11::init<const char *, size_t, const vector<int>&, bool>())
        .def("save", &sample_onnx::Engine::save)
        .def("infer", &sample_onnx::Engine::infer)
        // .def_property_readonly("input_size", &sample_onnx::Engine::getInputSize)
        .def_static("load", [](const string &path) {
            return new sample_onnx::Engine(path);
        })
        .def("__call__", [](sample_onnx::Engine &engine, at::Tensor data) {
            return infer(engine, data);
        });
}
