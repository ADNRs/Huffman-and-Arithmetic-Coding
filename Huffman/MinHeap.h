#ifndef __MIN_HEAP_H__
#define __MIN_HEAP_H__

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "MergeSort.h"

template <typename ValType> class MinHeap;

template <typename ValType>
class MinHeap<ValType *> {
    std::vector<ValType *> __array;
    std::vector<ValType *> &array;
    std::unordered_map<ValType *, uint64_t> map;

public:
    MinHeap();
    MinHeap(std::vector<ValType *> &);
    int64_t size() const;
    bool empty() const;
    ValType * get_top();
    void insert(ValType *);
    ValType * extract(uint64_t=0);
    void heapify(int);
    ValType * erase(ValType *);
    void clear();
    bool exist(ValType *);
    void resize(uint64_t=10);
    void reheapify();
};

template <typename ValType>
MinHeap<ValType *>::MinHeap() : array(__array) {}

// O(n)
template <typename ValType>
MinHeap<ValType *>::MinHeap(std::vector<ValType *> &array) : array(array) {
    for (uint64_t i = 0; i < array.size(); i++) {
        map[array[i]] = i;
    }

    reheapify();
}

template <typename ValType>
int64_t MinHeap<ValType *>::size() const {
    return array.size();
}

template <typename ValType>
bool MinHeap<ValType *>::empty() const {
    return array.empty();
}

// O(1)
template <typename ValType>
ValType * MinHeap<ValType *>::get_top() {
    return array[0];
}

// O(log n)
template <typename ValType>
void MinHeap<ValType *>::insert(ValType *val) {
    array.push_back(val);

    int64_t i = size() - 1;
    int64_t left;

    map[val] = i;

    while (i > 0 && smaller_than<ValType>(array[i], array[(left = (i - 1) / 2)])) {
        std::swap(map[array[left]], map[array[i]]);
        std::swap(array[left], array[i]);
        i = left;
    }
}

// O(log n)
template <typename ValType>
ValType * MinHeap<ValType *>::extract(uint64_t idx) {
    ValType *val = array[idx];
    array[idx] = array[array.size() - 1];

    map[array[idx]] = idx;
    map.erase(val);

    array.pop_back();
    heapify(idx);
    return val;
}

// O(log n)
template <typename ValType>
void MinHeap<ValType *>::heapify(int i) {
    int64_t j, left, right;

    while ((left = i*2 + 1) < size()) {
        right = i*2 + 2;
        j = right < size() && smaller_than<ValType>(array[right], array[left]) ? right : left;

        if (smaller_than<ValType>(array[j], array[i])) {
            std::swap(map[array[i]], map[array[j]]);
            std::swap(array[i], array[j]);
        }

        i = j;
    }
}

template <typename ValType>
ValType * MinHeap<ValType *>::erase(ValType *key) {
    return map.find(key) == map.end() ? nullptr : extract(map[key]);
}

template <typename ValType>
void MinHeap<ValType *>::clear() {
    array.clear();
    map.clear();
}

template <typename ValType>
bool MinHeap<ValType *>::exist(ValType *key) {
    return map.find(key) != map.end();
}

template <typename ValType>
void MinHeap<ValType *>::resize(uint64_t n) {
    array.resize(n);
    map.reserve(n);
    map.rehash(n);
}

template <typename ValType>
void MinHeap<ValType *>::reheapify() {
    for (int64_t i = array.size()/2 - 1; i >= 0; i--) {
        heapify(i);
    }
}

#endif
