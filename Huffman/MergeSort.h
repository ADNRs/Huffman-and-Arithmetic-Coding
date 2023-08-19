#ifndef __MERGE_SORT_H__
#define __MERGE_SORT_H__

#include <algorithm>
#include <cstdint>
#include <vector>

template <typename ValType>
bool smaller_than(const ValType * const lhs, const ValType * const rhs) {
    return *lhs < *rhs;
}

template <typename ValType>
void mergesort(std::vector<ValType *> &arr, uint64_t left, uint64_t right) {
    if (left >= right) return;

    if (right - left >= 16) {
        uint64_t mid = left + (right - left)/2;

        #pragma omp task shared(arr) untied if (right - left >= 8192)
        mergesort(arr, left, mid);
        #pragma omp task shared(arr) untied if (right - left >= 8192)
        mergesort(arr, mid + 1, right);
        #pragma omp taskwait
        std::inplace_merge(arr.begin() + left, arr.begin() + mid + 1, arr.begin() + right + 1, smaller_than<ValType>);
    }
    else {
        std::sort(arr.begin() + left, arr.begin() + right + 1, smaller_than<ValType>);
    }
}

template <typename ValType>
void mergesort(std::vector<ValType *> &arr) {
    #pragma omp parallel
    #pragma omp single
    mergesort(arr, 0, arr.size() - 1);
}

#endif
