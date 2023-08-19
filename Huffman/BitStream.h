#ifndef __BIT_STREAM_H__
#define __BIT_STREAM_H__

#include <cstdint>
#include <vector>

class BitStream {
    const std::vector<uint8_t> &buf;
    uint64_t idx;
    int8_t bidx;

public:
    BitStream(const std::vector<uint8_t> &);
    bool next();
    bool empty() const;
};

inline BitStream::BitStream(const std::vector<uint8_t> &buf) : buf(buf), idx(0), bidx(7) {}

inline bool BitStream::next() {
    if (bidx == -1) [[unlikely]] {
        bidx = 7;
        idx++;
    }

    return ((buf[idx] >> bidx--) & 0b00000001);
}

inline bool BitStream::empty() const {
    return idx == buf.size() - 1 && bidx == -1;
}

#endif
