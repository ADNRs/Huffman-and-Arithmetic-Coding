#ifndef __AC_ENCODER_H__
#define __AC_ENCODER_H__

#include <string>
#include "ProbabilityModel.h"
#include "SymbolStream.h"

template <typename SymbolType, typename StorageType, typename ProbModelType, uint64_t word_length, bool show_step=false>
class ACEncoder {
    static constexpr uint64_t get_offset() {
        constexpr uint64_t real_length = sizeof (StorageType) * 8;
        static_assert(word_length < real_length, "#bit of StorageType must be greater than word_length");

        return real_length - word_length;
    }

    static constexpr StorageType get_max_value() {
        return StorageType(-1) >> get_offset();
    }

    static constexpr StorageType get_valid_mask() {
        return get_max_value();
    }

    static constexpr StorageType get_half_value() {
        return (get_max_value() + 1) >> 1;
    }

    void correct_bound(StorageType &bound) {
        bound &= get_valid_mask();
    }

    void update_bounds(StorageType &lower_bound, StorageType &upper_bound, const Bound prob_bound) {
        StorageType interval = upper_bound - lower_bound + 1;

        upper_bound = lower_bound + StorageType(interval * prob_bound.upper) - 1;
        lower_bound = lower_bound + StorageType(interval * prob_bound.lower);

        correct_bound(lower_bound);
        correct_bound(upper_bound);
    }

    bool get_msb(StorageType bound) {
        return bound >= get_half_value();
    }

    void shift_bounds(StorageType &lower_bound, StorageType &upper_bound) {
        lower_bound <<= 1;
        upper_bound <<= 1;
        upper_bound |= 1;

        correct_bound(lower_bound);
        correct_bound(upper_bound);
    }

    void shift_bounds_e3(StorageType &lower_bound, StorageType &upper_bound) {
        lower_bound <<= 1;
        upper_bound <<= 1;
        upper_bound |= 1;

        lower_bound += get_half_value();
        upper_bound += get_half_value();

        correct_bound(lower_bound);
        correct_bound(upper_bound);
    }

    bool check_e3(StorageType lower_bound, StorageType upper_bound) {
        constexpr StorageType e3_lower_bound = get_half_value() >> 1;
        constexpr StorageType e3_upper_bound = get_half_value() | e3_lower_bound;

        return e3_lower_bound <= lower_bound && upper_bound < e3_upper_bound;
    }

    std::string get_binary_representation(StorageType bound) {
        std::string str;
        StorageType mask = get_half_value();
        // StorageType mask = (StorageType(-1) >> 1) + 1;

        while (mask > 0) {
            str += (bound & mask ? "1" : "0");
            mask >>= 1;
        }

        return str;
    }

public:
    uint64_t encode(BufferedSymbolStream<SymbolType> bss, ProbModelType prob_model, std::vector<uint8_t> chrs={}) {
        StorageType lower_bound = 0;
        StorageType upper_bound = get_max_value();
        uint64_t e3_count = 0;
        uint64_t msg_count = 0;

        uint64_t read_symbols = 0;

        std::string msg;

        if constexpr (show_step) {
            std::cout << "Initialization" << std::endl;
            std::cout << "            lower_bound=" << get_binary_representation(lower_bound)
                      << ", upper_bound=" << get_binary_representation(upper_bound)
                      << std::endl;
        }

        while (!bss.empty()) {
            std::vector<SymbolType> symbols = bss.next();
            Bounds bounds = prob_model.get_prob(symbols);

            read_symbols++;

            if constexpr (show_step) {
                std::cout << "Prefix=";
                if (chrs.size() > 0) {
                    for (uint64_t i = 0; i < symbols.size() - 1; i++) std::cout << chrs[symbols[i]];
                    std::cout << ", Symbol=" << chrs[symbols.back()] << std::endl;
                }
                else {
                    for (uint64_t i = 0; i < symbols.size() - 1; i++) std::cout << uint64_t(symbols[i]);
                    std::cout << ", Symbol=" << uint64_t(symbols.back()) << std::endl;
                }
            }

            for (uint64_t i = 0; i < bounds.size(); i++) {
                update_bounds(lower_bound, upper_bound, bounds[i]);

                if constexpr (show_step) {
                    if (i + 1 == bounds.size()) {
                        if (chrs.size()) {
                            std::cout << "    Encode " << chrs[symbols.back()] << std::endl;
                        }
                        else {
                            std::cout << "    Encode " << uint64_t(symbols.back()) << std::endl;
                        }
                    }
                    else {
                        std::cout << "    Encode <esc>" << std::endl;
                    }
                    std::cout << "        Update bounds with (" << bounds[i].lower << ", " << bounds[i].upper << ")"
                              << std::endl
                              << "            lower_bound=" << get_binary_representation(lower_bound)
                              << ", upper_bound=" << get_binary_representation(upper_bound)
                              << std::endl;
                }

                while (true) {
                    bool lower_msb = get_msb(lower_bound);
                    bool upper_msb = get_msb(upper_bound);

                    if (lower_msb == upper_msb) {
                        shift_bounds(lower_bound, upper_bound);

                        msg_count++;

                        if constexpr (show_step) {
                            if (lower_msb == 0) {
                                msg += "0";
                            }
                            else {
                                msg += "1";
                            }
                        }

                        while (e3_count > 0) {
                            if constexpr (show_step) {
                                if (lower_msb == 0) {
                                    msg += "1";
                                }
                                else {
                                    msg += "0";
                                }
                            }

                            msg_count++;
                            e3_count--;
                        }

                        if constexpr (show_step) {
                            if (lower_msb == 0) {
                                std::cout << "        e1 | msg=" << msg << std::endl;
                            }
                            else {
                                std::cout << "        e2 | msg=" << msg << std::endl;
                            }

                            std::cout << "            lower_bound=" << get_binary_representation(lower_bound)
                                      << ", upper_bound=" << get_binary_representation(upper_bound)
                                      << std::endl;
                        }
                    }
                    else if (check_e3(lower_bound, upper_bound)) {
                        shift_bounds_e3(lower_bound, upper_bound);

                        e3_count++;

                        if constexpr (show_step) {
                            std::cout << "        e3 | cnt=" << e3_count
                                      << std::endl
                                      << "            lower_bound=" << get_binary_representation(lower_bound)
                                      << ", upper_bound=" << get_binary_representation(upper_bound)
                                      << std::endl;
                        }
                    }
                    else {
                        break;
                    }
                }
            }

            prob_model.update(symbols);
        }

        if constexpr (show_step) {
            std::cout << "Length: " << msg.size() << std::endl;
            std::cout << "Message: " << msg << std::endl;
        }

        return msg_count;
    }
};

#endif
