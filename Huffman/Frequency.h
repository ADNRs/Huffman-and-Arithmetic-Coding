#ifndef __FREQUENCY_H__
#define __FREQUENCY_H__

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace std {
template <>
struct hash<__uint128_t> {
    size_t operator()(const __uint128_t &val) const {
        return std::hash<uint64_t>{}((uint64_t)val ^ (uint64_t)(val >> 64));
    }
};
}

template <typename KeyType, typename ValueType, uint64_t denom=10>
class Frequency {
    std::vector<ValueType> vec;
    std::unordered_map<KeyType, ValueType> map;
    std::vector<KeyType> nonzero_elems;

    ValueType & (Frequency<KeyType, ValueType, denom>::*access_impl)(KeyType);
    ValueType (Frequency<KeyType, ValueType, denom>::*get_impl)(KeyType);

    KeyType nelem;
    __uint128_t occurrence;

    ValueType & __access_map(KeyType);
    ValueType & __access_vec(KeyType);
    ValueType __get_map(KeyType);
    ValueType __get_vec(KeyType);

public:
    Frequency(KeyType);
    Frequency(const Frequency &);
    Frequency & operator=(const Frequency &);
    ValueType operator[](KeyType);
    ValueType & access(KeyType);
    ValueType get(KeyType);
    double get_freq(KeyType);
    void count(KeyType, __uint128_t=1);
    void count(KeyType, __uint128_t, __uint128_t);
    KeyType size() const;
    __uint128_t count_occurrence() const;
    KeyType count_nonzeros() const;
    std::vector<KeyType> & get_nonzero_elems();
    void clear();
};

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType & Frequency<KeyType, ValueType, denom>::__access_map(KeyType idx) {
    if (map.size() < nelem / denom) [[likely]] {
        auto it = map.find(idx);

        if (it == map.end()) {
            nonzero_elems.push_back(idx);
            return map[idx] = 0;
        }

        return it->second;
    }
    else {
        vec = std::vector<ValueType>(nelem, 0);

        for (auto &[key, val] : map) {
            vec[key] = val;
        }

        map.clear();

        access_impl = &Frequency<KeyType, ValueType, denom>::__access_vec;
        get_impl = &Frequency<KeyType, ValueType, denom>::__get_vec;

        return (this->*access_impl)(idx);
    }
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType & Frequency<KeyType, ValueType, denom>::__access_vec(KeyType idx) {
    if (vec[idx] == 0) [[unlikely]] {
        nonzero_elems.push_back(idx);
    }

    return vec[idx];
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType Frequency<KeyType, ValueType, denom>::__get_map(KeyType idx) {
    if (map.find(idx) == map.end()) {
        return 0;
    }
    else {
        return map[idx];
    }
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType Frequency<KeyType, ValueType, denom>::__get_vec(KeyType idx) {
    return vec[idx];
}

template <typename KeyType, typename ValueType, uint64_t denom>
Frequency<KeyType, ValueType, denom>::Frequency(KeyType nelem) :
    access_impl(&Frequency<KeyType, ValueType, denom>::__access_map),
    get_impl(&Frequency<KeyType, ValueType, denom>::__get_map),
    nelem(nelem),
    occurrence(0) {
    if constexpr (sizeof (KeyType) >= sizeof (uint64_t)) {
        if (nelem > KeyType{1} << 52) {
            map.reserve(25000000);
            map.rehash(25000000);
        }
        else if (nelem > KeyType{1} << 32) {
            map.reserve(10000000);
            map.rehash(10000000);
        }
        else {
            map.reserve(10000);
            map.rehash(10000);
        }
    }
}

template <typename KeyType, typename ValueType, uint64_t denom>
Frequency<KeyType, ValueType, denom>::Frequency(const Frequency &other) {
    vec = other.vec;
    map = other.map;
    nonzero_elems = other.nonzero_elems;
    access_impl = other.access_impl == &Frequency<KeyType, ValueType, denom>::__access_map ? &Frequency<KeyType, ValueType, denom>::__access_map : &Frequency<KeyType, ValueType, denom>::__access_vec;
    get_impl = other.get_impl == &Frequency<KeyType, ValueType, denom>::__get_map ? &Frequency<KeyType, ValueType, denom>::__get_map : &Frequency<KeyType, ValueType, denom>::__get_vec;
    nelem = other.nelem;
    occurrence = other.occurrence;
}

template <typename KeyType, typename ValueType, uint64_t denom>
Frequency<KeyType, ValueType, denom> & Frequency<KeyType, ValueType, denom>::operator=(const Frequency &other) {
    vec = other.vec;
    map = other.map;
    nonzero_elems = other.nonzero_elems;
    access_impl = other.access_impl == &Frequency<KeyType, ValueType, denom>::__access_map ? &Frequency<KeyType, ValueType, denom>::__access_map : &Frequency<KeyType, ValueType, denom>::__access_vec;
    get_impl = other.get_impl == &Frequency<KeyType, ValueType, denom>::__get_map ? &Frequency<KeyType, ValueType, denom>::__get_map : &Frequency<KeyType, ValueType, denom>::__get_vec;
    nelem = other.nelem;
    occurrence = other.occurrence;

    return *this;
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType Frequency<KeyType, ValueType, denom>::operator[](KeyType idx) {
    return get(idx);
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType & Frequency<KeyType, ValueType, denom>::access(KeyType idx) {
    return (this->*access_impl)(idx);
}

template <typename KeyType, typename ValueType, uint64_t denom>
ValueType Frequency<KeyType, ValueType, denom>::get(KeyType idx) {
    return (this->*get_impl)(idx);
}

template <typename KeyType, typename ValueType, uint64_t denom>
double Frequency<KeyType, ValueType, denom>::get_freq(KeyType idx) {
    return 1.0 * get(idx) / occurrence;
}

template <typename KeyType, typename ValueType, uint64_t denom>
void Frequency<KeyType, ValueType, denom>::count(KeyType idx, __uint128_t amount) {
    access(idx) += amount;
    occurrence += amount;
}

template <typename KeyType, typename ValueType, uint64_t denom>
void Frequency<KeyType, ValueType, denom>::count(KeyType idx, __uint128_t amount, __uint128_t occ_amount) {
    access(idx) += amount;
    occurrence += occ_amount;
}

template <typename KeyType, typename ValueType, uint64_t denom>
KeyType Frequency<KeyType, ValueType, denom>::size() const {
    return nelem;
}

template <typename KeyType, typename ValueType, uint64_t denom>
__uint128_t Frequency<KeyType, ValueType, denom>::count_occurrence() const {
    return occurrence;
}

template <typename KeyType, typename ValueType, uint64_t denom>
KeyType Frequency<KeyType, ValueType, denom>::count_nonzeros() const {
    return nonzero_elems.size();
}

template <typename KeyType, typename ValueType, uint64_t denom>
std::vector<KeyType> & Frequency<KeyType, ValueType, denom>::get_nonzero_elems() {
    return nonzero_elems;
}

template <typename KeyType, typename ValueType, uint64_t denom>
void Frequency<KeyType, ValueType, denom>::clear() {
    vec.clear();
    map.clear();
    nonzero_elems.clear();
    occurrence = 0;
}

#endif
