#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "ACEncoder.h"
#include "ProbabilityModel.h"
#include "SymbolStream.h"

std::vector<uint8_t> get_exercise_alphabets() {
    /*
        h, e, t, a, c, ∆
        0, 1, 2, 3, 4, 5
    */
    return {'h', 'e', 't', 'a', 'c', '_'};
}

std::vector<uint8_t> get_exercise_sequence() {
    /*
        c  a  t  ∆  a  t  e  ∆  h  a  t
        4, 3, 2, 5, 3, 2, 1, 5, 0, 3, 2
    */
    return {4, 3, 2, 5, 3, 2, 1, 5, 0, 3, 2};
}

void show_exercise_step() {
    std::vector<uint8_t> seq = get_exercise_sequence();
    std::vector<uint8_t> chrs = get_exercise_alphabets();
    constexpr uint64_t word_length = 6;
    constexpr uint64_t order = 1;
    uint8_t symbol_type;
    uint8_t storage_type;

    BufferedSymbolStream<decltype(symbol_type)> fixed_bss(seq, sizeof (symbol_type)*8, 1);
    BufferedSymbolStream<decltype(symbol_type)> ppma_bss(seq, sizeof (symbol_type)*8, order+1);

    FixedProbabilityModel<decltype(symbol_type), decltype(storage_type)>      fixed(6, fixed_bss);
    PPM<decltype(symbol_type), decltype(storage_type), PPM_Mode::ppma, false> ppma(chrs.size());

    ACEncoder<decltype(symbol_type), decltype(storage_type), decltype(fixed), word_length, true> fixed_enc;
    ACEncoder<decltype(symbol_type), decltype(storage_type), decltype(ppma), word_length, true> ppma_enc;

    fixed_enc.encode(fixed_bss, fixed, chrs);
    std::cout << std::endl;
    ppma_enc.encode(ppma_bss, ppma, chrs);
}

template <typename SymbolType, typename StorageType, uint64_t stride, uint64_t order, uint64_t nsymbols, uint64_t word_length>
void run_all_test(const std::vector<uint8_t> &sequence) {
    BufferedSymbolStream<SymbolType> fixed_bss(sequence, stride, 1);
    BufferedSymbolStream<SymbolType> ppm_bss(sequence, stride, order+1);

    FixedProbabilityModel<SymbolType, StorageType>      fixed(nsymbols, fixed_bss);
    PPM<SymbolType, StorageType, PPM_Mode::ppma, false> ppman(nsymbols);
    PPM<SymbolType, StorageType, PPM_Mode::ppma, true>  ppmae(nsymbols);
    PPM<SymbolType, StorageType, PPM_Mode::ppmb, false> ppmbn(nsymbols);
    PPM<SymbolType, StorageType, PPM_Mode::ppmb, true>  ppmbe(nsymbols);
    PPM<SymbolType, StorageType, PPM_Mode::ppmc, false> ppmcn(nsymbols);
    PPM<SymbolType, StorageType, PPM_Mode::ppmc, true>  ppmce(nsymbols);

    ACEncoder<SymbolType, StorageType, decltype(fixed), word_length, false> fixed_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppman), word_length, false> ppman_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppmae), word_length, false> ppmae_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppmbn), word_length, false> ppmbn_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppmbe), word_length, false> ppmbe_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppmcn), word_length, false> ppmcn_enc;
    ACEncoder<SymbolType, StorageType, decltype(ppmce), word_length, false> ppmce_enc;

    uint64_t cnts[7] = {};

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        {
            if constexpr (stride < 16 || (stride < 32 && order <= 2)) {
                #pragma omp task
                cnts[0] = fixed_enc.encode(fixed_bss, fixed);

                #pragma omp task
                cnts[1] = ppman_enc.encode(ppm_bss, ppman);

                #pragma omp task
                cnts[2] = ppmae_enc.encode(ppm_bss, ppmae);

                #pragma omp task
                cnts[3] = ppmbn_enc.encode(ppm_bss, ppmbn);

                #pragma omp task
                cnts[4] = ppmbe_enc.encode(ppm_bss, ppmbe);

                #pragma omp task
                cnts[5] = ppmcn_enc.encode(ppm_bss, ppmcn);

                #pragma omp task
                cnts[6] = ppmce_enc.encode(ppm_bss, ppmce);
            }
            else if constexpr (stride == 32 && order == 3) {
                #pragma omp task
                {
                    cnts[0] = fixed_enc.encode(fixed_bss, fixed);
                    cnts[1] = ppman_enc.encode(ppm_bss, ppman);
                }

                #pragma omp task
                {
                    cnts[2] = ppmae_enc.encode(ppm_bss, ppmae);
                    cnts[3] = ppmbn_enc.encode(ppm_bss, ppmbn);
                }

                #pragma omp task
                {
                    cnts[4] = ppmbe_enc.encode(ppm_bss, ppmbe);
                    cnts[5] = ppmcn_enc.encode(ppm_bss, ppmcn);
                }

                #pragma omp task
                cnts[6] = ppmce_enc.encode(ppm_bss, ppmce);
            }
            else {
                #pragma omp task
                {
                    cnts[0] = fixed_enc.encode(fixed_bss, fixed);
                    cnts[1] = ppman_enc.encode(ppm_bss, ppman);
                    cnts[2] = ppmae_enc.encode(ppm_bss, ppmae);
                    cnts[3] = ppmbn_enc.encode(ppm_bss, ppmbn);
                }

                #pragma omp task
                {
                    cnts[4] = ppmbe_enc.encode(ppm_bss, ppmbe);
                    cnts[5] = ppmcn_enc.encode(ppm_bss, ppmcn);
                    cnts[6] = ppmce_enc.encode(ppm_bss, ppmce);
                }
            }
        }
    }

    std::chrono::duration<double> elapsed_time = std::chrono::high_resolution_clock::now() - start_time;

    std::cout << "stride=" << stride << ", order=" << order << ", nsymbols=" << nsymbols
              << ", word_length=" << word_length << std::endl;
    std::cout << "    Fixed : " << cnts[0] << " bits" << std::endl;
    std::cout << "    PPMA  : " << cnts[1] << " bits" << std::endl;
    std::cout << "    PPMAe : " << cnts[2] << " bits" << std::endl;
    std::cout << "    PPMB  : " << cnts[3] << " bits" << std::endl;
    std::cout << "    PPMBe : " << cnts[4] << " bits" << std::endl;
    std::cout << "    PPMC  : " << cnts[5] << " bits" << std::endl;
    std::cout << "    PPMCe : " << cnts[6] << " bits" << std::endl;
    std::cout << "Time: " << elapsed_time.count() << " seconds" << std::endl;
}

void test_exercise() {
    std::vector<uint8_t> seq = get_exercise_sequence();

    std::cout << "Using \'cat∆cat∆hat\' with ";
    run_all_test<uint8_t, uint8_t, 8, 1, 6, 6>(seq);
}

void test_repeated_sequence() {
    std::vector<uint8_t> seq;

    for (int i = 0; i < 48; i++) {
        seq.push_back(i % 4);
    }

    std::cout << "Using \'";
    for (auto &s : seq) std::cout << int(s);
    std::cout << "\' with ";

    run_all_test<uint8_t, uint8_t, 8, 2, 4, 6>(seq);
}

void test_random_sequence() {
    std::vector<uint8_t> seq;

    for (int i = 0; i < 48; i++) {
        seq.push_back(rand() % 4);
    }

    std::cout << "Using \'";
    for (auto &s : seq) std::cout << int(s);
    std::cout << "\' with ";

    run_all_test<uint8_t, uint8_t, 8, 2, 4, 6>(seq);
}

template <uint64_t stride, uint64_t order>
void test_alexnet() {
    std::fstream f{"./alexnet.pth", std::ios::in|std::ios::binary};

    if (f.fail()) throw "./alexnet path not found";

    std::vector<uint8_t> seq = {std::istreambuf_iterator<char>(f), {}};

    std::cout << "Using \'alexnet\' with ";
    run_all_test<uint64_t, uint64_t, stride, order, 1ULL << stride, sizeof (uint64_t)*8 - 1>(seq);
}

template <uint64_t stride, uint64_t max_order>
void test_alexnet_stride() {
    if constexpr (max_order >= 0) {
        test_alexnet<stride, 0>();
        std::cout << std::endl;
    }

    if constexpr (max_order >= 1) {
        test_alexnet<stride, 1>();
        std::cout << std::endl;
    }

    if constexpr (max_order >= 2) {
        test_alexnet<stride, 2>();
        std::cout << std::endl;
    }

    if constexpr (max_order >= 3) {
        test_alexnet<stride, 3>();
        std::cout << std::endl;
    }

    if constexpr (max_order >= 4) {
        test_alexnet<stride, 4>();
        std::cout << std::endl;
    }
}

int main() {
    show_exercise_step();
    std::cout << std::endl;

    test_exercise();
    std::cout << std::endl;

    test_random_sequence();
    std::cout << std::endl;

    test_repeated_sequence();
    std::cout << std::endl;

    test_alexnet_stride<1, 4>();
    test_alexnet_stride<2, 4>();
    test_alexnet_stride<4, 4>();
    test_alexnet_stride<8, 4>();
    test_alexnet_stride<16, 3>();
    test_alexnet_stride<32, 2>();

    return 0;
}
