#ifndef __PROB_MODEL_H__
#define __PROB_MODEL_H__

#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "SymbolStream.h"

struct Bound {
    double lower;
    double upper;
};

template <typename type>
struct VectorHasher {
    int32_t operator()(const std::vector<type> &vec) const {
        int32_t hash = vec.size();

        for (auto &i : vec) {
            hash ^= i + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }

        return hash;
    }
};

using Bounds = std::vector<Bound>;

enum PPM_Mode {none, ppma, ppmb, ppmc};

template <typename SymbolType, typename StorageType, uint8_t ppm_mode>
struct PPMContext {
    std::unordered_map<SymbolType, uint64_t> symbol2index;
    std::vector<StorageType> cum_count;
    StorageType esc_count;
    void (PPMContext::*update_impl)(const SymbolType);

    PPMContext() : esc_count(0) {
        symbol2index.reserve(100);
        symbol2index.rehash(100);
        cum_count.reserve(100);

        cum_count.push_back(0);

        if constexpr (ppm_mode == PPM_Mode::none) {
            update_impl = &PPMContext::none;
        }
        else if constexpr (ppm_mode == PPM_Mode::ppma) {
            update_impl = &PPMContext::ppma;
        }
        else if constexpr (ppm_mode == PPM_Mode::ppmb) {
            update_impl = &PPMContext::ppmb;
        }
        else if constexpr (ppm_mode == PPM_Mode::ppmc) {
            update_impl = &PPMContext::ppmc;
        }
        else {
            throw "unknown PPM mode";
        }
    }

    bool find(const SymbolType symbol) const {
        return symbol2index.find(symbol) != symbol2index.end();
    }

    StorageType get_tot_count() const {
        return cum_count.back() + esc_count;
    }

    Bound get_bound(const SymbolType symbol) const {
        StorageType tot_count = get_tot_count();
        uint64_t index = symbol2index.at(symbol);

        return {double(cum_count[index-1]) / tot_count, double(cum_count[index]) / tot_count};
    }

    Bound get_bound(const SymbolType symbol, const std::unordered_set<SymbolType> &exclusion_symbols) const {
        StorageType tot_count = get_tot_count();
        uint64_t index = symbol2index.at(symbol);
        StorageType exclusion_count = 0;

        for (const auto &s : exclusion_symbols) {
            uint64_t s_index = symbol2index.at(s);

            if (s_index < index) {
                exclusion_count += cum_count[s_index] - cum_count[s_index - 1];
            }
        }

        tot_count -= exclusion_count;

        return {double(cum_count[index-1] - exclusion_count) / tot_count, double(cum_count[index] - exclusion_count) / tot_count};
    }

    Bound get_esc_bound() const {
        return {double(cum_count.back()) / get_tot_count(), 1};
    }

    std::unordered_set<SymbolType> get_appeared_symbols() const {
        std::unordered_set<SymbolType> symbols;

        for (auto &[k, v] : symbol2index) {
            symbols.insert(k);
        }

        return symbols;
    }

    StorageType get_exclusion_count(const std::unordered_set<SymbolType> symbols) const {
        StorageType exclusion_count = 0;

        for (auto &s : symbols) {
            if (find(s)) {
                SymbolType index = symbol2index.at(s);
                exclusion_count += cum_count[index] - cum_count[index - 1];
            }
        }

        return exclusion_count;
    }

    void update(const SymbolType symbol) {
        (this->*update_impl)(symbol);
    }

    void none(const SymbolType symbol) {
        if (!find(symbol)) {
            symbol2index[symbol] = cum_count.size();
            cum_count.push_back(cum_count.back());
        }

        for (uint64_t index = symbol2index[symbol]; index < cum_count.size(); index++) {
            cum_count[index]++;
        }
    }

    void ppma(const SymbolType symbol) {
        if (!find(symbol)) {
            if (esc_count == 0) {
                esc_count = 1;
            }

            symbol2index[symbol] = cum_count.size();
            cum_count.push_back(cum_count.back());
        }

        for (uint64_t index = symbol2index[symbol]; index < cum_count.size(); index++) {
            cum_count[index]++;
        }
    }

    void ppmb(const SymbolType symbol) {
        if (!find(symbol)) {
            esc_count += 1;
            symbol2index[symbol] = cum_count.size();
            cum_count.push_back(cum_count.back());
            return;
        }

        for (uint64_t index = symbol2index[symbol]; index < cum_count.size(); index++) {
            cum_count[index]++;
        }
    }

    void ppmc(const SymbolType symbol) {
        if (!find(symbol)) {
            esc_count += 1;
            symbol2index[symbol] = cum_count.size();
            cum_count.push_back(cum_count.back());
        }

        for (uint64_t index = symbol2index[symbol]; index < cum_count.size(); index++) {
            cum_count[index]++;
        }
    }
};

template <typename SymbolType, typename StorageType, uint8_t ppm_mode>
class PPMContexts {
    std::unordered_map<std::vector<SymbolType>, PPMContext<SymbolType, StorageType, ppm_mode>, VectorHasher<SymbolType>> contexts;

public:
    PPMContexts() {
        contexts.reserve(100000);
        contexts.rehash(100000);
    }

    bool find(const std::vector<SymbolType> prefix) const {
        return contexts.find(prefix) != contexts.end();
    }

    const PPMContext<SymbolType, StorageType, ppm_mode> &get_context(const std::vector<SymbolType> prefix) const {
        return contexts.at(prefix);
    }

    PPMContext<SymbolType, StorageType, ppm_mode> &get_context(const std::vector<SymbolType> prefix) {
        if (!find(prefix)) {
            contexts[prefix] = {};
        }

        return contexts[prefix];
    }
};

template <typename SymbolType, typename StorageType>
class BaseProbabilityModel {
    uint64_t nsymbols;

public:
    BaseProbabilityModel(const uint64_t nsymbols) : nsymbols(nsymbols) {}
    virtual Bounds get_prob(const std::vector<SymbolType> symbols) const;
    virtual void update(const std::vector<SymbolType> symbols);
    uint64_t get_nsymbols() const { return nsymbols; }
};

template <typename SymbolType, typename StorageType>
class FixedProbabilityModel : public BaseProbabilityModel<SymbolType, StorageType> {
    using BaseModel = BaseProbabilityModel<SymbolType, StorageType>;
    PPMContext<SymbolType, StorageType, PPM_Mode::none> prob;

public:
    FixedProbabilityModel(const uint64_t nsymbols, BufferedSymbolStream<SymbolType> bss) : BaseModel(nsymbols) {
        while (!bss.empty()) {
            prob.update(bss.next().back());
        }
    }

    virtual Bounds get_prob(const std::vector<SymbolType> symbols) const override {
        return {prob.get_bound(symbols[0])};
    }

    virtual void update(const std::vector<SymbolType>) override {
        return;
    }
};

template <typename SymbolType, typename StorageType, uint8_t ppm_mode, bool use_exclusion>
class PPM : public BaseProbabilityModel<SymbolType, StorageType> {
    using BaseModel = BaseProbabilityModel<SymbolType, StorageType>;
    PPMContexts<SymbolType, StorageType, ppm_mode> contexts;

public:
    PPM(const uint64_t nsymbols) : BaseModel(nsymbols) {}

    virtual Bounds get_prob(const std::vector<SymbolType> symbols) const override {
        Bounds bounds;
        const SymbolType symbol = symbols.back();
        std::unordered_set<SymbolType> exclusion_symbols;

        for (int64_t order = symbols.size() - 1; order >= -1; order--) {
            if (order == -1) {
                bounds.emplace_back(1.0 * symbol / BaseModel::get_nsymbols(), (1.0*symbol + 1) / BaseModel::get_nsymbols());
            }
            else {
                std::vector<SymbolType> prefix(symbols.begin() + symbols.size() - order - 1, symbols.end() - 1);

                if (contexts.find(prefix)) {
                    const PPMContext<SymbolType, StorageType, ppm_mode> &context = contexts.get_context(prefix);

                    if (context.find(symbol)) {
                        if constexpr (ppm_mode == PPM_Mode::ppmb) {
                            Bound bound;

                            if constexpr (use_exclusion) {
                                bound = context.get_bound(symbol, exclusion_symbols);
                            }
                            else {
                                bound = context.get_bound(symbol);
                            }

                            if (bound.lower == bound.upper) {
                                bounds.push_back(context.get_esc_bound());
                            }
                            else {
                                bounds.push_back(bound);
                                break;
                            }
                        }
                        else {
                            Bound bound;

                            if constexpr (use_exclusion) {
                                bound = context.get_bound(symbol, exclusion_symbols);
                            }
                            else {
                                bound = context.get_bound(symbol);
                            }

                            bounds.push_back(bound);
                            break;
                        }
                    }
                    else {
                        bounds.push_back(context.get_esc_bound());
                    }

                    if constexpr (use_exclusion) {
                        for (const SymbolType &s : context.get_appeared_symbols()) {
                            exclusion_symbols.insert(s);
                        }
                    }
                }
            }
        }

        return bounds;
    }

    virtual void update(const std::vector<SymbolType> symbols) override {
        const SymbolType symbol = symbols.back();

        for (uint64_t order = 0; order < symbols.size(); order++) {
            std::vector<SymbolType> prefix(symbols.begin() + symbols.size() - order - 1, symbols.end() - 1);

            contexts.get_context(prefix).update(symbol);
        }
    }
};

#endif
