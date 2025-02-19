#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <vector>
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    

void GraphObj::optimize() {
    if (!this->topo_sort()) {
        return;
    }

    bool optimized;
    do {
        optimized = false;

        for (size_t i = 0; i < ops.size(); ++i) {
            auto op = ops[i];
            if (op->getOpType() == OpType::Transpose) {
                auto opd = std::dynamic_pointer_cast<TransposeObj>(op);
                auto input = op->getInputs(0);
                auto prevOp = input->getSource();

                if (prevOp && prevOp->getOpType() == OpType::Transpose && input->getTargets().size() == 1) {
                    auto prevOpd = std::dynamic_pointer_cast<TransposeObj>(prevOp);
                    auto prevInput = prevOp->getInputs(0);

                    // Combine permutations
                    auto perm = opd->getPermute();
                    bool isIdentity = true;
                    for (size_t j = 0; j < perm.size(); ++j) {
                        perm[j] = prevOpd->getPermute()[perm[j]];
                        if (perm[j] != int(j)) {
                            isIdentity = false;
                        }
                    }

                    prevInput->removeTarget(prevOp);
                    if (isIdentity) {
                        for (auto succ : op->getSuccessors()) {
                            succ->replaceInput(op->getOutput(), prevInput);
                            prevInput->addTarget(succ);
                        }
                        this->removeTensor(op->getOutput());
                    } else {
                        auto newOp = make_ref<TransposeObj>(this, prevInput, op->getOutput(), perm);
                        this->addOperatorAndConnect(newOp);
                    }

                    for (auto pred : prevOp->getPredecessors()) {
                        pred->removeSuccessors(prevOp);
                    }
                    for (auto succ : op->getSuccessors()) {
                        succ->removePredecessors(op);
                    }

                    
                    this->removeTensor(input);
                    this->removeOperator(op);
                    this->removeOperator(prevOp);
                    optimized = true;
                    i -= 2;  
                    break;
                }
            }

            if (op->getOpType() == OpType::MatMul) {
                auto matmulOp = std::dynamic_pointer_cast<MatmulObj>(op);

                // Check inputs for transpose
                for (int inputIdx = 0; inputIdx < 2; ++inputIdx) {
                    auto input = op->getInputs(inputIdx);
                    auto transposeOp = input->getSource();

                    if (transposeOp && transposeOp->getOpType() == OpType::Transpose && input->getTargets().size() == 1) {
                        auto transposeObj = std::dynamic_pointer_cast<TransposeObj>(transposeOp);
                        auto perm = transposeObj->getPermute();

                        bool isLastTwoSwapped = (perm.size() > 1 && perm[perm.size() - 2] == int(perm.size() - 1) &&
                                                 perm[perm.size() - 1] == int(perm.size() - 2));
                        bool isIdentityElsewhere = true;
                        for (size_t j = 0; j < perm.size() - 2; ++j) {
                            if (perm[j] != int(j)) {
                                isIdentityElsewhere = false;
                                break;
                            }
                        }
                        if (!isLastTwoSwapped || !isIdentityElsewhere) {
                            continue;
                        }

                        // Merge transpose into MatMul
                        if (inputIdx == 0) {
                            matmulOp->setTransA(!matmulOp->getTransA());
                        } else {
                            matmulOp->setTransB(!matmulOp->getTransB());
                        }

                        // Update connections
                        auto prevInput = transposeOp->getInputs(0);
                        prevInput->removeTarget(transposeOp);
                        prevInput->addTarget(matmulOp);
                        matmulOp->replaceInput(input, prevInput);
                        matmulOp->removePredecessors(transposeOp);

                        // Remove Transpose
                        this->removeTensor(input);
                        this->removeOperator(transposeOp);
                        optimized = true;
                        break;
                    }
                }
            }
        }
    } while (optimized);
}







    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        size_t sizeTensorPtr = this->tensors.size();
        std::vector<size_t> offset;
        // getBytes()
        for(const auto &tensor:tensors){
            offset.push_back(allocator.alloc(tensor->getBytes()));
        }
        auto dptr = this->allocator.getPtr();
        for(size_t i = 0 ;i<sizeTensorPtr;++i){
            auto rptr = reinterpret_cast<char*>(dptr) + offset[i];
            this->tensors[i] -> setDataBlob(make_ref<BlobObj>(this->runtime, (void*)rptr));
        }
        
        allocator.info();


    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
