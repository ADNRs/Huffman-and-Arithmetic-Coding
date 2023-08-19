#ifndef __SYMBOL_STREAM_H__
#define __SYMBOL_STREAM_H__

#include <cstdint>
#include <vector>

class BitStream {
    const std::vector<uint8_t> &buf;
    uint64_t idx;
    int8_t bidx;

public:
    BitStream(const std::vector<uint8_t> &buf) : buf(buf), idx(0), bidx(7) {}

    bool next() {
        if (bidx == -1) [[unlikely]] {
            bidx = 7;
            idx++;
        }

        return ((buf[idx] >> bidx--) & 0b00000001);
    }

    bool empty() const {
        return idx == buf.size() - 1 && bidx == -1;
    }
};

template <typename SymbolType>
class SymbolStream {
    BitStream bs;
    uint64_t stride;

public:
    SymbolStream(const std::vector<uint8_t> &buf, uint64_t stride) : bs(buf), stride(stride) {}

    SymbolType next() {
        SymbolType symbol = 0;

        for (uint64_t i = stride; i > 0; i--) {
            if (bs.empty()) [[unlikely]] {
                return symbol << i;
            }

            symbol <<= 1;
            symbol |= bs.next();
        }

        return symbol;
    }

    bool empty() const {
        return bs.empty();
    }
};

template <typename SymbolType>
class BufferedSymbolStream {
    SymbolStream<SymbolType> ss;
    uint64_t size;
    std::vector<SymbolType> symbols;

public:
    BufferedSymbolStream(const std::vector<uint8_t> &buf, uint64_t stride, uint64_t size) : ss(buf, stride), size(size) {}

    std::vector<SymbolType> next() {
        SymbolType symbol = ss.next();

        if (symbols.size() == size) {
            symbols.erase(symbols.begin());
        }

        symbols.push_back(symbol);

        return symbols;
    }

    bool empty() const {
        return ss.empty();
    }
};

#endif
