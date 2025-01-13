#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        // 获取输入张量 A 和 B 的形状
        const auto A = inputs[0];
        const auto B = inputs[1];
        const auto shapeA = A->getDims(); 
        const auto shapeB = B->getDims(); 

        int rankA = shapeA.size();
        int rankB = shapeB.size();
        if (rankA < 2 || rankB < 2)
        {return std::nullopt;} // 输入张量维度必须至少为 2


        Shape transposedShapeA = shapeA;
        Shape transposedShapeB = shapeB;
        if (transA)
        {
        std::swap(transposedShapeA[rankA - 1], transposedShapeA[rankA - 2]);
        }
        if (transB)
        {
        std::swap(transposedShapeB[rankB - 1], transposedShapeB[rankB - 2]);
        }

        int K1 = transposedShapeA[rankA - 1];
        int K2 = transposedShapeB[rankB - 2];
        if (K1 != K2)
            {return std::nullopt;}


        Shape outputShape;
        // 批量维度部分（除了最后两个维度）
        for (int i = 0; i < std::max(rankA, rankB) - 2; ++i)
        {
            int dimA = (i < rankA - 2) ? transposedShapeA[i] : 1;
            int dimB = (i < rankB - 2) ? transposedShapeB[i] : 1;
            outputShape.push_back(std::max(dimA, dimB));
        }

        int M = transposedShapeA[rankA - 2];
        int N = transposedShapeB[rankB - 1];
        outputShape.push_back(M);
        outputShape.push_back(N);

        return vector<Shape>{outputShape};
    }

} // namespace infini
