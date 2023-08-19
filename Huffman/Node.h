#ifndef __TREE_NODE_H__
#define __TREE_NODE_H__

#include <cstdint>
#include <type_traits>

template <typename ValueType>
struct Node {
    ValueType freq;
    Node *left;
    Node *right;

    Node(ValueType freq, Node *left=nullptr, Node *right=nullptr) : freq(freq), left(left), right(right) {}
    virtual ~Node() = default;

    friend bool operator<(const Node<ValueType> &lhs, const Node<ValueType> &rhs) {
        return lhs.freq < rhs.freq;
    }
};

template <typename KeyType, typename ValueType>
struct LeafNode : public Node<ValueType> {
    KeyType tag;

    LeafNode(KeyType tag, ValueType freq) : Node<ValueType>::Node(freq), tag(tag) {}
};

template <typename KeyType, typename ValueType>
struct AdaptiveNode : public LeafNode<KeyType, ValueType> {
    KeyType id;
    AdaptiveNode *parent;

    AdaptiveNode(KeyType id, KeyType tag, ValueType freq, AdaptiveNode *parent=nullptr) :
        LeafNode<KeyType, ValueType>::LeafNode(tag, freq),
        id(id),
        parent(parent) {}

    friend bool operator<(const AdaptiveNode<KeyType, ValueType> &lhs, const AdaptiveNode<KeyType, ValueType> &rhs) {
        return lhs.id > rhs.id;
    }
};

#endif
