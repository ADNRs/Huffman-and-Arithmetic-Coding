#ifndef __ALPHABET_STREAM_H__
#define __ALPHABET_STREAM_H__

#include <cstdint>
#include <vector>

#include "BitStream.h"

template <typename KeyType>
class AlphabetStream {
    BitStream bit_stream;
    uint64_t stride;

public:
    AlphabetStream(const std::vector<uint8_t> &, uint64_t);
    KeyType next();
    bool empty() const;
};

template <typename KeyType>
inline AlphabetStream<KeyType>::AlphabetStream(const std::vector<uint8_t> &buf, uint64_t stride) : bit_stream(buf), stride(stride) {}

template <typename KeyType>
inline KeyType AlphabetStream<KeyType>::next() {
    KeyType alphabet = 0;

    for (uint64_t i = stride; i > 0; i--) {
        if (bit_stream.empty()) [[unlikely]] {
            return alphabet << i;
        }

        alphabet <<= 1;
        alphabet |= bit_stream.next();
    }

    return alphabet;
}

template <typename KeyType>
inline bool AlphabetStream<KeyType>::empty() const {
    return bit_stream.empty();
}

#endif
