#ifndef __EXTENDED_HUFFMAN_H__
#define __EXTENDED_HUFFMAN_H__

#include <algorithm>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>

#include "AlphabetStream.h"
#include "Frequency.h"
#include "MergeSort.h"
#include "MinHeap.h"
#include "Node.h"

template <typename KeyType, typename ValueType, bool par_read=false, bool par_build=false, uint64_t extend_size=1>
class ExtendedHuffman {
    Frequency<KeyType, ValueType> freq;
    uint64_t stride;
    std::chrono::duration<double> elapsed_time;
    __uint128_t encoded_size;

    void build_freq(const std::vector<uint8_t> &buf) {
        if constexpr (par_read) {
            uint64_t lcm = std::lcm(8, stride);
            uint64_t step = lcm * (1 * 1024 * 1024 / lcm);

            #pragma omp parallel for schedule(dynamic, 1)
            for (uint64_t i = 0; i < buf.size() / step; i++) {
                std::vector<uint8_t>::const_iterator start = buf.begin() + i*step;
                std::vector<uint8_t>::const_iterator end = buf.begin() + (i + 1)*step;

                std::vector<uint8_t> sub_buf{start, end};
                AlphabetStream<KeyType> data{{sub_buf}, stride};
                Frequency<KeyType, ValueType> temp{KeyType{1} << stride};

                for (uint64_t j = 0; j < step * 8 / stride; j++) {
                    temp.count(data.next());
                }

                #pragma omp critical
                for (auto &a : temp.get_nonzero_elems()) {
                    freq.count(a, temp[a]);
                }
            }

            if (buf.size() / step * step < buf.size()) {
                std::vector<uint8_t>::const_iterator start = buf.begin() + (buf.size() / step * step);
                std::vector<uint8_t>::const_iterator end = buf.begin() + buf.size();

                std::vector<uint8_t> sub_buf{start, end};
                AlphabetStream<KeyType> data{sub_buf, stride};

                while (!data.empty()) {
                    freq.count(data.next());
                }
            }
        }
        else {
            AlphabetStream<KeyType> data{buf, stride};

            while (!data.empty()) {
                freq.count(data.next());
            }
        }

        if constexpr (extend_size > 1) {
            Frequency<KeyType, ValueType> base_freq = freq;

            for (uint64_t i = 2; i <= extend_size; i++) {
                Frequency<KeyType, ValueType> temp_freq{KeyType{1} << (stride * i)};

                for (auto &extend_key : freq.get_nonzero_elems()) {
                    for (auto &base_key : base_freq.get_nonzero_elems()) {
                        auto new_key = (extend_key << stride) | base_key;
                        temp_freq.count(new_key, freq[extend_key] * base_freq[base_key]);
                    }
                }

                freq = temp_freq;
            }
        }
    }

    void build_coding_table() {
        if constexpr (par_build) {
            auto nonzeros = freq.get_nonzero_elems();

            std::vector<Node<ValueType> *> leaf_nodes(nonzeros.size(), nullptr);
            std::vector<Node<ValueType> *> internal_nodes;

            #pragma omp parallel for schedule(dynamic, 100000)
            for (uint64_t i = 0; i < nonzeros.size(); i++) {
                leaf_nodes[i] = new LeafNode<KeyType, ValueType>{nonzeros[i], freq[nonzeros[i]]};
            }

            mergesort(leaf_nodes);

            uint64_t leaf_ptr = 0;
            uint64_t internal_ptr = 0;

            while ((leaf_nodes.size() - leaf_ptr) + (internal_nodes.size() - internal_ptr) > 1) {
                Node<ValueType> *node[2];

                #pragma GCC unroll 2
                for (uint32_t i = 0; i < 2; i++) {
                    if (internal_ptr == internal_nodes.size()) [[unlikely]] {
                        node[i] = leaf_nodes[leaf_ptr++];
                    }
                    else if (leaf_ptr == leaf_nodes.size()) [[unlikely]] {
                        node[i] = internal_nodes[internal_ptr++];
                    }
                    else [[likely]] {
                        if (*leaf_nodes[leaf_ptr] < *internal_nodes[internal_ptr]) {
                            node[i] = leaf_nodes[leaf_ptr++];
                        }
                        else {
                            node[i] = internal_nodes[internal_ptr++];
                        }
                    }
                }

                internal_nodes.push_back(new Node<ValueType>{node[0]->freq + node[1]->freq, node[0], node[1]});
            }

            Node<ValueType> *root = leaf_ptr < leaf_nodes.size() ? leaf_nodes[leaf_ptr] : internal_nodes[internal_ptr];

            #pragma omp parallel
            #pragma omp single
            traverse(root);

            #pragma omp parallel for schedule(dynamic, 100000)
            for (uint64_t i = 0; i < leaf_nodes.size(); i++) {
                delete leaf_nodes[i];
            }

            #pragma omp parallel for schedule(dynamic, 100000)
            for (uint64_t i = 0; i < internal_nodes.size(); i++) {
                delete internal_nodes[i];
            }
        }
        else {
            std::vector<Node<ValueType> *> nodes;
            std::vector<Node<ValueType> *> free_nodes;

            for (auto &key : freq.get_nonzero_elems()) {
                Node<ValueType> *node = new LeafNode<KeyType, ValueType>{key, freq[key]};

                nodes.push_back(node);
                free_nodes.push_back(node);
            }

            MinHeap<Node<ValueType> *> heap{nodes};

            while (heap.size() > 1) {
                Node<ValueType> *node = heap.extract();
                Node<ValueType> *node2 = heap.extract();
                Node<ValueType> *new_node = new Node<ValueType>{node->freq + node2->freq, node, node2};

                heap.insert(new_node);
                free_nodes.push_back(new_node);
            }

            Node<ValueType> *root = heap.extract();

            #pragma omp parallel
            #pragma omp single
            traverse(root);

            #pragma omp parallel for schedule(dynamic, 100000)
            for (uint64_t i = 0; i < free_nodes.size(); i++) {
                delete free_nodes[i];
            }
        }
    }

    void traverse(Node<ValueType> *root, uint64_t codeword_length=0) {
        if (root->left) {
            #pragma omp task
            traverse(root->left, codeword_length + 1);
        }

        if (root->right) {
            #pragma omp task
            traverse(root->right, codeword_length + 1);
        }

        if (root->left == root->right) {
            const auto &alphabet = dynamic_cast<LeafNode<KeyType, ValueType> *>(root)->tag;
            #pragma omp critical
            encoded_size += codeword_length * freq[alphabet];
        }
    }

public:
    ExtendedHuffman(const std::vector<uint8_t> &buf, uint64_t stride) : freq(Frequency<KeyType, ValueType>{KeyType{1} << stride}),
    stride(stride), encoded_size(0) {
        auto start_time = std::chrono::high_resolution_clock::now();

        build_freq(buf);
        build_coding_table();

        elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
    }

    __uint128_t get_nonzeros() const {
        return freq.count_nonzeros();
    }

    double get_expected_codeword_length() {
        return 1.0 * encoded_size / freq.count_occurrence();
    }

    double get_compression_ratio() {
        return 1.0 * freq.count_occurrence() * stride * extend_size / encoded_size;
    }

    double get_execution_time() const {
        return elapsed_time.count();
    }

    __uint128_t get_occurrence() const {
        return freq.count_occurrence();
    }

    void dump() {
        auto n  = get_nonzeros();
        auto o  = get_occurrence();
        auto cl = get_expected_codeword_length();
        auto cr = get_compression_ratio();
        auto t  = get_execution_time();

        std::cout << "Extended Symbol Width:    " << stride << " * " << extend_size << " = " << stride * extend_size << " (bit)" << std::endl;
        std::cout << "Nonzero Symbols:          " << n                       << std::endl;
        std::cout << "Effective Data Size:      " << o      << " (# symbol)" << std::endl;
        std::cout << "Expected Codeword Length: " << cl     << " (bit)"      << std::endl;
        std::cout << "Compression Ratio:        " << cr                      << std::endl;
        std::cout << "Execution Time:           " << t      << " (second)"   << std::endl;
    }

    std::map<KeyType, double> get_PMF() {
        std::map<KeyType, double> pmf;

        for (auto &key : freq.get_nonzero_elems()) {
            pmf[key] = freq.get_freq(key);
        }

        return pmf;
    }
};

#endif
