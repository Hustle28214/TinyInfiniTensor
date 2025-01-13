#include "core/allocator.h"
#include <utility>
#include <list>
#include <algorithm>
#include <cmath>
namespace infini
{
    
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
        
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        // 对齐大小
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================


        // first fit
        // 从可用内存块的开始位置顺序查找，找到第一个满足需求的内存块。
        // 分配该内存块，如果有剩余空间，则将剩余部分标记为新的可用块。
        auto it = freeBlockMap.begin();
        for (; it != freeBlockMap.end(); ++it) {
            if (it->second >= size) {
                size_t start = it->first;

                used += size;
                peak = std::max(used, peak);

                size_t remain = it->second - size;
                if (remain > 0) {
                    freeBlockMap[start + size] = remain;
                }

                freeBlockMap.erase(it);

                return start;
            }
        }

        // 如果没有找到，重新分配内存
        size_t new_start = used;
        used += size;
        peak = std::max(peak, used);

        return new_start;

    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
    
        used -= size;

        // 将回收的内存块插入到 freeBlockMap 中
        freeBlockMap[addr] = size;

        // 合并相邻的空闲内存块
        auto it = freeBlockMap.find(addr);
        if (it != freeBlockMap.end()) {
            auto prev_it = it;
            if (prev_it != freeBlockMap.begin()) {
                --prev_it;
                if (prev_it->first + prev_it->second == addr) {
                    prev_it->second += size;
                    freeBlockMap.erase(it);
                    it = prev_it;
                }
            }

            auto next_it = it;
            ++next_it;
            if (next_it != freeBlockMap.end() && addr + size == next_it->first) {
                it->second += next_it->second;
                freeBlockMap.erase(next_it);
            }
        }
        
        
        }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}

