#ifndef __ADAPTIVE_HUFFMAN_H__
#define __ADAPTIVE_HUFFMAN_H__

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "AlphabetStream.h"
#include "Block.h"
#include "Frequency.h"
#include "Node.h"

template <typename KeyType, typename ValueType, bool progress=false, bool debug=false, bool block_opt=true>
class AdaptiveHuffman {
    AdaptiveNode<KeyType, ValueType> *root;
    AdaptiveNode<KeyType, ValueType> *NTY;

    std::unordered_map<KeyType, AdaptiveNode<KeyType, ValueType> *> node_list;
    BlockRecorder<KeyType, ValueType> block;
    Frequency<KeyType, ValueType> len_count;
    Frequency<KeyType, ValueType> freq;

    uint64_t stride;
    KeyType next_id;
    uint64_t encoded_size;
    std::chrono::duration<double> elapsed_time;

    // nalpha = 2^e + r
    const uint64_t e;
    const uint64_t r;

public:
    AdaptiveHuffman(const std::vector<uint8_t> &buf, uint64_t stride, KeyType nalpha, uint64_t e, uint64_t r=0) :
    root(nullptr), len_count(nalpha), freq(nalpha), stride(stride), next_id(nalpha - KeyType{1} + nalpha), encoded_size(0), e(e), r(r) {
        root = NTY = gen_node();

        if constexpr (sizeof (KeyType) >= sizeof (uint64_t)) {
            if (nalpha > KeyType{1} << 52) {
                node_list.reserve(25000000);
                node_list.rehash(25000000);
            }
            else if (nalpha > KeyType{1} << 32) {
                node_list.reserve(10000000);
                node_list.rehash(10000000);
            }
            else {
                node_list.reserve(10000);
                node_list.rehash(10000);
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        build_coding_table(buf);

        elapsed_time = std::chrono::high_resolution_clock::now() - start_time;
    }

    ~AdaptiveHuffman() {
        delete_node(root);
    }

    void delete_node(AdaptiveNode<KeyType, ValueType> *root) {
        if (root->left) {
            delete_node(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->left));
        }

        if (root->right) {
            delete_node(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->right));
        }

        delete root;
    }

    AdaptiveNode<KeyType, ValueType> * gen_node(KeyType tag=0, ValueType freq=0, AdaptiveNode<KeyType, ValueType> *parent=nullptr) {
        return new AdaptiveNode<KeyType, ValueType>{get_next_id(), tag, freq, parent};
    }

    KeyType get_next_id() {
        return next_id--;
    }

    uint64_t get_NTY_code_length(const KeyType &k) const {
        return e + (0 <= k && k < 2 * r);
    }

    std::string get_binary_representation(KeyType v, uint64_t len) const {
        std::string ret;

        for (uint64_t i = len; i > 0; i--) {
            ret += std::to_string((v >> (i - 1)) & 1);
        }

        return ret;
    }

    std::string get_NTY_code(const KeyType &k) const {
        const uint64_t len = get_NTY_code_length(k);

        if (0 <= k && k < 2 * r) {
            return get_binary_representation(k, len);
        }
        else {
            return get_binary_representation(k - r, len);
        }
    }

    uint64_t get_code_length(AdaptiveNode<KeyType, ValueType> *node) {
        uint64_t code_length = 0;

        if (!node) return 0;

        while (node->parent) {
            code_length++;
            node = node->parent;
        }

        return code_length;
    }

    std::string get_code(AdaptiveNode<KeyType, ValueType> *node) {
        std::string ret;

        while (node->parent) {
            ret += node->parent->left == node ? "0" : "1";
            node = node->parent;
        }

        std::reverse(ret.begin(), ret.end());

        return ret;
    }

    void swap_nodes(AdaptiveNode<KeyType, ValueType> *node1, AdaptiveNode<KeyType, ValueType> *node2) {
        if constexpr (debug) {
            dump_tree(root);
            std::cout << std::endl;
        }

        std::swap(node1->id, node2->id);

        if (node1->parent->left == node1) {
            if (node2->parent->left == node2) {
                std::swap(node1->parent->left, node2->parent->left);
            }
            else {
                std::swap(node1->parent->left, node2->parent->right);
            }
        }
        else {
            if (node2->parent->left == node2) {
                std::swap(node1->parent->right, node2->parent->left);
            }
            else {
                std::swap(node1->parent->right, node2->parent->right);
            }
        }

        std::swap(node1->parent, node2->parent);
    }

    void update(KeyType alpha) {
        auto it = node_list.find(alpha);
        AdaptiveNode<KeyType, ValueType> *curr_node = nullptr;

        if (it == node_list.end()) {
            AdaptiveNode<KeyType, ValueType> *node = gen_node(alpha, 1, NTY);
            AdaptiveNode<KeyType, ValueType> *new_NTY = gen_node(0, 0, NTY);

            node_list[alpha] = node;

            NTY->freq++;
            NTY->left = new_NTY;
            NTY->right = node;

            if constexpr (block_opt) {
                if (NTY != root) [[likely]] block.update(NTY);
                block.update(node);
            }

            curr_node = new_NTY->parent = node->parent = NTY;

            NTY = new_NTY;
        }
        else {
            curr_node = it->second;

            again:
            if constexpr (block_opt) {
                // get_max
                if (curr_node != root) [[likely]] {
                    AdaptiveNode<KeyType, ValueType> *max_node = block.get(curr_node);

                    // dump_tree(root);
                    // std::cout << "freq=" << curr_node->freq << ">> curr: " << std::to_string(curr_node->id) << ", max: " << std::to_string(max_node->id) << ", find: " << std::to_string(find_max_id_of_block(root, curr_node)->id) << std::endl;

                    if (curr_node->id < max_node->id && curr_node->parent != max_node) {
                        AdaptiveNode<KeyType, ValueType> *node1 = curr_node;
                        AdaptiveNode<KeyType, ValueType> *node2 = max_node;

                        swap_nodes(node1, node2);

                        block.reheapify(node1);
                    }

                    block.remove(curr_node);
                    curr_node->freq++;
                    block.update(curr_node);
                }
                else {
                    curr_node->freq++;
                }
            }
            else {
                AdaptiveNode<KeyType, ValueType> *max_node = find_max_id_of_block(root, curr_node);

                if (curr_node->id < max_node->id && curr_node->parent != max_node) {
                    AdaptiveNode<KeyType, ValueType> *node1 = curr_node;
                    AdaptiveNode<KeyType, ValueType> *node2 = max_node;

                    swap_nodes(node1, node2);
                }

                curr_node->freq++;
            }
        }

        if (curr_node->parent) {
            curr_node = curr_node->parent;
            goto again;
        }
    }

    AdaptiveNode<KeyType, ValueType> * find_max_id_of_block(AdaptiveNode<KeyType, ValueType> *root, AdaptiveNode<KeyType, ValueType> *target) {
        if (!root) return target;

        if (root->freq > target->freq) {
            AdaptiveNode<KeyType, ValueType> *left = find_max_id_of_block(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->left), target);
            AdaptiveNode<KeyType, ValueType> *right = find_max_id_of_block(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->right), target);

            return left->id > right->id ? left : right;
        }
        else if (root->freq == target->freq) {
            return root->id > target->id ? root : target;
        }
        else {
            return target;
        }
    }

    void build_coding_table(const std::vector<uint8_t> &buf) {
        AlphabetStream<KeyType> data{buf, stride};
        uint64_t cnt = 0;

        while (!data.empty()) {
            KeyType alpha = data.next();

            if constexpr (progress) {
                if (cnt++ % 1145 == 919) [[unlikely]] {
                    std::printf("\rProgress: %.2f%%", (double)cnt / ((buf.size() * 8.0) / stride) * 100);
                }
            }

            auto it = node_list.find(alpha);

            if constexpr (debug) {
                std::cout << (char)(alpha + 'a') << ": ";
            }

            if (it == node_list.end()) {
                ValueType len = get_code_length(NTY) + get_NTY_code_length(alpha);

                len_count.count(alpha, len, 1);
                freq.count(alpha);

                encoded_size += len;

                if constexpr (debug) {
                    std::cout <<  get_code(NTY) + get_NTY_code(alpha) << ", len=" << get_code_length(NTY) + get_NTY_code_length(alpha) << std::endl;
                }
            }
            else {
                ValueType len = get_code_length(it->second);

                len_count.count(alpha, len, 1);
                freq.count(alpha);

                encoded_size += len;

                if constexpr (debug) {
                    std::cout << get_code(it->second) << ", len=" << get_code_length(it->second) << std::endl;
                }
            }

            update(alpha);

            if constexpr (debug) {
                dump_tree(root);
                std::cout << std::endl;
            }
        }

        if constexpr (progress) {
            std::cout << "\r";
        }
    }

    uint64_t get_nonzeros() const {
        return freq.count_nonzeros();
    }

    double get_compression_ratio() {
        return 1.0 * freq.count_occurrence() * stride / encoded_size;
    }

    double get_execution_time() const {
        return elapsed_time.count();
    }

    uint64_t get_occurrence() const {
        return freq.count_occurrence();
    }

    void dump() {
        auto n  = get_nonzeros();
        auto o  = freq.count_occurrence();
        auto cl = get_expected_codeword_length();
        auto cr = get_compression_ratio();
        auto t  = get_execution_time();

        std::cout << "Symbol Length:            " << stride << " (bit)"      << std::endl;
        std::cout << "Nonzero Symbol:           " << n                       << std::endl;
        std::cout << "Data Size:                " << o      << " (# symbol)" << std::endl;
        std::cout << "Expected Codeword Length: " << cl     << " (bit)"      << std::endl;
        std::cout << "Compression Ratio:        " << cr                      << std::endl;
        std::cout << "Execution Time:           " << t      << " (second)"   << std::endl;
    }

    double get_expected_codeword_length() {
        double codeword = 0;

        for (auto &alpha : len_count.get_nonzero_elems()) {
            codeword += len_count.get_freq(alpha);
        }

        return codeword;
    }

    void dump_tree(AdaptiveNode<KeyType, ValueType> *root, std::string indent="", bool is_the_last=true) {
        if (!root) return;

        if (root == NTY) {
            std::cout << indent << "+- <NTY: " << std::to_string(root->id) << "/" << std::to_string(root->freq) << ">" << std::endl;
        }
        else if (root->left || root->right) {
            std::cout << indent << "+- <internal: " << std::to_string(root->id) << "/" << std::to_string(root->freq) << ">" << std::endl;
        }
        else {
            std::cout << indent << "+- <tag: " << std::to_string(root->id) << "/" << std::to_string(root->freq) << ">:" << (char)(root->tag + 'a') << std::endl;
        }

        indent += is_the_last ? "   " : "|  ";

        dump_tree(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->right), indent, false);
        dump_tree(dynamic_cast<AdaptiveNode<KeyType, ValueType> *>(root->left), indent, true);
    }

    std::map<KeyType, double> get_average_codeword_length_per_alphabet() {
        std::map<KeyType, double> pmf;

        for (auto &key : len_count.get_nonzero_elems()) {
            pmf[key] = (double)len_count[key] / freq[key];
        }

        return pmf;
    }
};

#endif
