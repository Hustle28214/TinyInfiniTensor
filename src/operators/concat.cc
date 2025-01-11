#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================

    // IT_ASSERT: All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_dims = inputs[i]->getDims();
        IT_ASSERT(input_dims.size() == rank);
        for (size_t j = 0; j < rank; ++j) {
            if (j != static_cast<size_t>(dim) && input_dims[j] != dims[j]) {
                IT_ASSERT(false);// the datatype of 'dim' must be transformed into size_t
            }
        }
    }

    size_t totalDims = 0;
    for (const auto &input : inputs) {
        auto targetDim = input->getDims()[dim];
        totalDims += targetDim;
    }

    dims[dim] = totalDims;
    return {{dims}};

}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
