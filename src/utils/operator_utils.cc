#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    // Shape getDims() const { return shape; }
    // auto res = infer_broadcast(A->getDims(), B->getDims());
    
    size_t rankA = A.size();
    size_t rankB = B.size();
    size_t rankMax = std::max(rankA, rankB);

    Shape extendedA(rankMax, 1);
    Shape extendedB(rankMax, 1);
    Shape broadcastShape(rankMax);

    
    for (size_t i = 0; i < rankA; ++i) {
        extendedA[rankMax - rankA + i] = A[i];
    }
    for (size_t i = 0; i < rankB; ++i) {
        extendedB[rankMax - rankB + i] = B[i];
    }

    for (size_t i = 0; i < rankMax; ++i) {
        if (extendedA[i] == extendedB[i]) {
            broadcastShape[i] = extendedA[i];
        } else if (extendedA[i] == 1) {
            broadcastShape[i] = extendedB[i];
        } else if (extendedB[i] == 1) {
            broadcastShape[i] = extendedA[i];
        } else {
            IT_ASSERT(false && "Incompatible shapes for broadcasting");
        }
    }

    return broadcastShape;
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini


} // namespace infini
