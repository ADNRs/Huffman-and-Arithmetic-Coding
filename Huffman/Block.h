#ifndef __BLOCK_H__
#define __BLOCK_H__

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#include "MinHeap.h"
#include "Node.h"

template <typename KeyType, typename ValueType>
class BlockRecorder {
    using NodeType = AdaptiveNode<KeyType, ValueType>;
    std::unordered_map<KeyType, MinHeap<NodeType *>> map;

public:
    void update(NodeType *node) {
        assert(node != nullptr);
        assert(map.find(node->freq) == map.end() || (map.find(node->freq) != map.end() && !map[node->freq].exist(node)));

        map[node->freq].insert(node);

        // std::cerr << "Update " << std::to_string(node->id) << " with freq=" << node->freq << ", curr max is " << std::to_string(map[node->freq].get_top()->id) << std::endl;
    }

    NodeType * get(NodeType *node) {
        assert(node != nullptr);
        assert(map.find(node->freq) != map.end());
        assert(map[node->freq].get_top()->id >= node->id);

        // std::cerr << "Get " << std::to_string(map[node->freq].get_top()->id) << " with freq=" << node->freq << " from node " << std::to_string(node->id) << std::endl;

        return map[node->freq].get_top();
    }

    void reheapify(NodeType *node) {
        assert(node != nullptr);
        assert(map.find(node->freq) != map.end());

        map[node->freq].reheapify();
    }

    NodeType * erase(NodeType *node) {
        assert(node != nullptr);
        assert(map.find(node->freq) != map.end());
        assert(map[node->freq].exist(node));

        // std::cerr << "Erase " << std::to_string(node->id) << " from freq=" << node->freq << std::endl;

        return map[node->freq].erase(node);
    }

    void remove(NodeType *node) {
        assert(node != nullptr);
        assert(map.find(node->freq) != map.end());
        assert(map[node->freq].exist(node));

        // std::cerr << "Remove " << std::to_string(node->id) << " from freq=" << node->freq << std::endl;

        if (map[node->freq].size() == 1) {
            if (node->freq < 1000) {
                map[node->freq].clear();
            }
            else {
                map.erase(node->freq);
            }
        }
        else {
            map[node->freq].erase(node);
        }
    }
};

#endif
