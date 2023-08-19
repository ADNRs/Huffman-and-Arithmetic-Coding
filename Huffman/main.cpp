#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "AdaptiveHuffman.h"
#include "ExtendedHuffman.h"
#include "Huffman.h"
#include "Node.h"

#ifdef PLOT
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#define IMAGE_PATH "./images/"
#endif

void print_header(std::string info) {
    std::cout << std::string(info.size() + 4, '*') << std::endl;
    std::cout << "* " << info << " *" << std::endl;
    std::cout << std::string(info.size() + 4, '*') << std::endl;
}

template <typename KeyType=uint64_t, typename ValueType=uint64_t>
void whole_data_experiment(const std::vector<uint8_t> &buf, uint64_t bit_width) {
    print_header(std::to_string(bit_width) + "-bit data source");
    Huffman<KeyType, ValueType, true, true> huf{buf, bit_width};
    huf.dump();
    std::cout << std::endl;

    #ifdef PLOT
    {
    std::vector<uint64_t> x;
    std::vector<double> y;

    for (auto &[a, f] : huf.get_PMF()) {
        x.push_back(a);
        y.push_back(f);
    }

    plt::clf();
    plt::figure_size(640, 480);
    plt::plot(x, y);
    plt::xlabel("Symbol");
    plt::ylabel("Probability");
    plt::title("PMF of " + std::to_string(bit_width) + "-Bit Data Source (Whole Data)");
    plt::save(IMAGE_PATH "PMF_" + std::to_string(bit_width) + "_BIT");
    }
    #endif
}

template <typename KeyType=uint64_t, typename ValueType=uint64_t>
void n_bit_experiment(const std::vector<uint8_t> &buf, uint64_t bit_width, uint64_t data_byte) {
    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 480);
    plt::xlabel("Symbol");
    plt::ylabel("Probability");
    plt::title("PMFs of " + std::to_string(bit_width) + "-Bit Data Source (" + std::to_string(data_byte) + "MB)");
    #endif
    for (uint64_t i = 0; i < buf.size(); i += data_byte * 1024 * 1024) {
        uint64_t start_MB = i / (1024 * 1024);
        uint64_t end_MB = std::min(start_MB + data_byte, buf.size() / 1024 / 1024);

        print_header(std::to_string(bit_width) + "-bit data source " + std::to_string(start_MB) + "MB-" + std::to_string(end_MB) + "MB");
        Huffman<KeyType, ValueType, true, true> huf{{buf.begin() + i, buf.begin() + std::min(i + data_byte*1024*1024, buf.size())}, bit_width};
        huf.dump();
        std::cout << std::endl;

        #ifdef PLOT
        std::vector<uint64_t> x;
        std::vector<double> y;

        for (auto &[a, f] : huf.get_PMF()) {
            x.push_back(a);
            y.push_back(f);
        }

        plt::named_plot(std::to_string(start_MB) + "MB-" + std::to_string(end_MB) + "MB", x, y, ":");
        #endif
    }
    #ifdef PLOT
    plt::legend();
    plt::save(IMAGE_PATH "PMFS_" + std::to_string(bit_width) + "_BIT_" + std::to_string(data_byte) + "MB.png");
    #endif
}

template <typename KeyType=__uint128_t, typename ValueType=uint64_t>
void speed_test(const std::vector<uint8_t> &buf) {
    constexpr uint64_t nbit = 64;

    std::vector<int> x(nbit, 0);
    std::vector<double> noopt_len(nbit, 0);
    std::vector<double> opt_len(nbit, 0);
    std::vector<double> noopt_time(nbit, 0);
    std::vector<double> opt_time(nbit, 0);

    for (uint64_t i = 1; i <= nbit; i++) {
        Huffman<KeyType, ValueType, false, false> noopt_huf{buf, i};
        Huffman<KeyType, ValueType, true, true> opt_huf{buf, i};

        noopt_len[i - 1] = noopt_huf.get_expected_codeword_length();
        opt_len[i - 1] = opt_huf.get_expected_codeword_length();
        noopt_time[i - 1] = noopt_huf.get_execution_time();
        opt_time[i - 1] = opt_huf.get_execution_time();
        x[i - 1] = i;
    }

    print_header("Huffman Speed Test: Naive vs Optimized");

    for (uint64_t i = 0; i < nbit; i++) {
        if (std::to_string(i + 1).size() == 1) {
            std::printf("Symbol Length = %lu               Naive       Optimized\n", i + 1);
        }
        else {
            std::printf("Symbol Length = %lu              Naive       Optimized\n", i + 1);
        }
        std::printf("Expected Codeword Length (bit)  %.6f    %.6f\n", noopt_len[i], opt_len[i]);
        std::printf("Execution Time (second)         %.6f    %.6f\n", noopt_time[i], opt_time[i]);
        std::printf("\n");
    }

    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 480);
    plt::named_plot("Naive", x, noopt_time);
    plt::named_plot("Optimized", x, opt_time);
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Execution Time (s)");
    plt::legend();
    plt::title("Huffman Speed Test: Naive vs Optimized");
    plt::save(IMAGE_PATH "speed_test");
    #endif
}

void width_experiment(const std::vector<uint8_t> &buf) {
    constexpr uint64_t nbit = 127;
    #ifdef PLOT
    std::vector<uint64_t> x(nbit, 0);
    std::vector<double> cl(nbit, 0);
    std::vector<double> cr(nbit, 0);
    std::vector<double> t(nbit, 0);
    std::vector<double> n(nbit, 0);
    std::vector<double> nr(nbit, 0);
    #endif
    #pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for (uint64_t i = 1; i <= nbit; i++) {
        Huffman<__uint128_t, uint64_t, false, true> huf{buf, i};
        print_header(std::to_string(i) + "-bit data source");
        huf.dump();
        std::cout << std::endl;

        #ifdef PLOT
        x[i - 1]  = i;
        cl[i - 1] = huf.get_expected_codeword_length();
        cr[i - 1] = huf.get_compression_ratio();
        t[i - 1]  = huf.get_execution_time();
        n[i - 1]  = huf.get_nonzeros();
        nr[i - 1] = (double)huf.get_nonzeros() / ((__uint128_t)1 << i);
        #endif
    }
    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 720);
    plt::subplot(3, 1, 1);
    plt::plot(x, cl);
    plt::title("Expected Codeword Length");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Expected Codeword Length (bit)");

    plt::subplot(3, 1, 2);
    plt::plot(x, cr);
    plt::title("Compression Ratio");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Compression Ratio");

    plt::subplot(3, 1, 3);
    plt::plot(x, t);
    plt::title("Execution Time");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Execution TIme (s)");

    // plt::subplot(5, 1, 4);
    // plt::plot(x, n);
    // plt::title("Nonzero Symbols");
    // plt::xlabel("Symbol Length (bit)");
    // plt::ylabel("Nonzero Symbols (# symbol)");

    // plt::subplot(5, 1, 5);
    // plt::plot(x, nr);
    // plt::title("Nonzero Symbols Ratio");
    // plt::xlabel("Symbol Length (bit)");
    // plt::ylabel("Nonzero Symbols Ratio");

    plt::suptitle("Effect of Different Symbol Lengths (__uint128)");
    plt::tight_layout();

    plt::save(IMAGE_PATH "symlen_1_127.png");
    #endif
}

void adaptive_huffman_textbook_NTY_code_test() {
    print_header("Textbook NTY Coding Test");
    std::vector<uint8_t> buf{'\0'};
    AdaptiveHuffman<uint8_t, uint64_t> huf{buf, 8, 26, 4, 10};

    std::cout << "NTY Code for \'a\'" << std::endl;
    std::cout << "Expected: 00000, Returned: " << huf.get_NTY_code('a'-'a') << std::endl;
    std::cout << "NTY Code for \'b\'" << std::endl;
    std::cout << "Expected: 00001, Returned: " << huf.get_NTY_code('b'-'a') << std::endl;
    std::cout << "NTY Code for \'v\'" << std::endl;
    std::cout << "Expected: 1011, Returned: " << huf.get_NTY_code('v'-'a') << std::endl;
    std::cout << std::endl;
}

void adaptive_huffman_textbook_example() {
    print_header("Textbook Encoding Example");
    std::vector<uint8_t> buf{'a' - 'a', 'a' - 'a', 'r' - 'a', 'd' - 'a', 'v' - 'a'};
    AdaptiveHuffman<uint8_t, uint64_t, false, true> huf{buf, 8, 26, 4, 10};
    std::cout << std::endl;
}

template <typename KeyType=uint64_t, typename ValueType=uint64_t>
void adaptive_huffman_whole_data_experiment(const std::vector<uint8_t> &buf, uint64_t bit_width) {
    print_header("AdaHuff: " + std::to_string(bit_width) + "-bit data source");
    AdaptiveHuffman<KeyType, ValueType, false, false, true> huf{buf, bit_width, KeyType{1} << bit_width, bit_width};
    huf.dump();
    std::cout << std::endl;

    #ifdef PLOT
    {
    std::vector<uint64_t> x;
    std::vector<double> y;

    for (auto &[a, f] : huf.get_average_codeword_length_per_alphabet()) {
        x.push_back(a);
        y.push_back(f);
    }

    plt::clf();
    plt::figure_size(640, 480);
    plt::plot(x, y, ",");
    plt::xlabel("Symbol");
    plt::ylabel("Average Codeword Length (bit)");
    plt::title("Avg. Codeword Length of Each Symbol of " + std::to_string(bit_width) + "-Bit Data Source");
    plt::save(IMAGE_PATH "AH_PMF_" + std::to_string(bit_width) + "_BIT");
    }
    #endif
}

template <typename KeyType=uint64_t, typename ValueType=uint64_t>
void adaptive_huffman_n_bit_experiment(const std::vector<uint8_t> &buf, uint64_t bit_width, uint64_t data_byte) {
    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 480);
    plt::xlabel("Symbol");
    plt::ylabel("Average Codeword Length");
    plt::title("Avg. Codeword Length of Each Symbol of " + std::to_string(bit_width) + "-Bit Data Source (" + std::to_string(data_byte) + "MB)");
    #endif
    for (uint64_t i = 0; i < buf.size(); i += data_byte * 1024 * 1024) {
        uint64_t start_MB = i / (1024 * 1024);
        uint64_t end_MB = std::min(start_MB + data_byte, buf.size() / 1024 / 1024);

        print_header("AdaHuff: " + std::to_string(bit_width) + "-bit data source " + std::to_string(start_MB) + "MB-" + std::to_string(end_MB) + "MB");
        AdaptiveHuffman<KeyType, ValueType, false, false, true> huf{{buf.begin() + i, buf.begin() + std::min(i + data_byte*1024*1024, buf.size())}, bit_width, KeyType{1} << bit_width, bit_width};
        huf.dump();
        std::cout << std::endl;

        #ifdef PLOT
        std::vector<uint64_t> x;
        std::vector<double> y;

        for (auto &[a, f] : huf.get_average_codeword_length_per_alphabet()) {
            x.push_back(a);
            y.push_back(f);
        }

        plt::named_plot(std::to_string(start_MB) + "MB-" + std::to_string(end_MB) + "MB", x, y, ",");
        #endif
    }
    #ifdef PLOT
    plt::legend();
    plt::save(IMAGE_PATH "AH_PMFS_" + std::to_string(bit_width) + "_BIT_" + std::to_string(data_byte) + "MB.png");
    #endif
}

template <typename KeyType=uint64_t, typename ValueType=uint64_t>
void adaptive_huffman_speed_test(const std::vector<uint8_t> &buf) {
    constexpr uint64_t nbit = 10;

    std::vector<int> x(nbit, 0);
    std::vector<double> noopt_len(nbit, 0);
    std::vector<double> opt_len(nbit, 0);
    std::vector<double> noopt_time(nbit, 0);
    std::vector<double> opt_time(nbit, 0);

    for (uint64_t i = 1; i <= nbit; i++) {
        AdaptiveHuffman<KeyType, ValueType, false, false, false> noopt_huf{buf, i, KeyType{1} << i, i};
        AdaptiveHuffman<KeyType, ValueType, false, false, true> opt_huf{buf, i, KeyType{1} << i, i};

        noopt_len[i - 1] = noopt_huf.get_expected_codeword_length();
        opt_len[i - 1] = opt_huf.get_expected_codeword_length();
        noopt_time[i - 1] = noopt_huf.get_execution_time();
        opt_time[i - 1] = opt_huf.get_execution_time();
        x[i - 1] = i;
    }

    print_header("Adaptive Huffman Speed Test: Naive vs Optimized");

    for (uint64_t i = 0; i < nbit; i++) {
        if (std::to_string(i + 1).size() == 1) {
            std::printf("Symbol Length = %lu               Naive       Optimized\n", i + 1);
        }
        else {
            std::printf("Symbol Length = %lu              Naive       Optimized\n", i + 1);
        }
        std::printf("Expected Codeword Length (bit)  %.6f    %.6f\n", noopt_len[i], opt_len[i]);
        std::printf("Execution Time (second)         %.6f    %.6f\n", noopt_time[i], opt_time[i]);
        std::printf("\n");
    }

    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 480);
    plt::named_plot("Naive", x, noopt_time);
    plt::named_plot("Optimized", x, opt_time);
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Execution Time (s)");
    plt::legend();
    plt::title("Adaptive Huffman Speed Test: Naive vs Optimized");
    plt::save(IMAGE_PATH "adahuff_speed_test");
    #endif
}

template <typename KeyType=__uint128_t, typename ValueType=uint64_t>
void adaptive_huffman_width_experiment(const std::vector<uint8_t> &buf) {
    constexpr uint64_t nbit = 24;
    #ifdef PLOT
    std::vector<uint64_t> x(nbit, 0);
    std::vector<double> cl(nbit, 0);
    std::vector<double> cr(nbit, 0);
    std::vector<double> t(nbit, 0);
    #endif
    #pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for (uint64_t i = 1; i <= nbit; i++) {
        AdaptiveHuffman<KeyType, ValueType> huf{buf, i, KeyType{1} << i, i};
        print_header("AdaHuff: " + std::to_string(i) + "-bit data source");
        huf.dump();
        std::cout << std::endl;

        #ifdef PLOT
        x[i - 1]  = i;
        cl[i - 1] = huf.get_expected_codeword_length();
        cr[i - 1] = huf.get_compression_ratio();
        t[i - 1]  = huf.get_execution_time();
        #endif
    }
    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 720);
    plt::subplot(3, 1, 1);
    plt::plot(x, cl);
    plt::title("Expected Codeword Length");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Expected Codeword Length (bit)");

    plt::subplot(3, 1, 2);
    plt::plot(x, cr);
    plt::title("Compression Ratio");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Compression Ratio");

    plt::subplot(3, 1, 3);
    plt::plot(x, t);
    plt::title("Execution Time");
    plt::xlabel("Symbol Length (bit)");
    plt::ylabel("Execution TIme (s)");

    plt::suptitle("AdaHuff: Effect of Different Symbol Lengths (__uint128)");
    plt::tight_layout();

    plt::save(IMAGE_PATH "AH_WIDTH_1_127.png");
    #endif
}

void extended_huffman(const std::vector<uint8_t> &buf) {
    std::vector<int> x8(3, 0);
    std::vector<double> cr8(3, 0);
    std::vector<int> x16(2, 0);
    std::vector<double> cr16(2, 0);
    std::vector<int> x32(2, 0);
    std::vector<double> cr32(2, 0);
    print_header("Extended Huffman: 8-Bit Experiment");
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 1> huf1{buf, 8};
    huf1.dump();
    std::cout << std::endl;
    x8[0] = 1;
    cr8[0] = huf1.get_compression_ratio();
    }
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 2> huf2{buf, 8};
    huf2.dump();
    std::cout << std::endl;
    x8[1] = 2;
    cr8[1] = huf2.get_compression_ratio();
    }
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 3> huf3{buf, 8};
    huf3.dump();
    std::cout << std::endl;
    x8[2] = 3;
    cr8[2] = huf3.get_compression_ratio();
    }
    print_header("Extended Huffman: 16-Bit Experiment");
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 1> huf1{buf, 16};
    huf1.dump();
    std::cout << std::endl;
    x16[0] = 1;
    cr16[0] = huf1.get_compression_ratio();
    }
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 2> huf2{buf, 16};
    huf2.dump();
    std::cout << std::endl;
    x16[1] = 2;
    cr16[1] = huf2.get_compression_ratio();
    }
    print_header("Extended Huffman: 32-Bit Experiment");
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 1> huf1{buf, 32};
    huf1.dump();
    std::cout << std::endl;
    x32[0] = 1;
    cr32[0] = huf1.get_compression_ratio();
    }
    {
    ExtendedHuffman<__uint128_t, __uint128_t, true, true, 2> huf2{buf, 32};
    huf2.dump();
    std::cout << std::endl;
    x32[1] = 2;
    cr32[1] = huf2.get_compression_ratio();
    }

    #ifdef PLOT
    plt::clf();
    plt::figure_size(640, 480);
    plt::named_plot("8-bit", x8, cr8, "-o");
    plt::named_plot("16-bit", x16, cr16, "-o");
    plt::named_plot("32-bit", x32, cr32, "-o");
    plt::xlabel("Extended Size");
    plt::ylabel("Compression Ratio");
    plt::legend();
    plt::title("Compression Ratio of Different Extended Size");
    plt::save(IMAGE_PATH "ext_huff");
    #endif
}

int main() {
    #ifdef _OPENMP
    omp_set_nested(1);
    #endif

    /***********************************************************/
    /* Read data                                               */
    /***********************************************************/
    std::fstream f{"./alexnet.pth", std::ios::in|std::ios::binary};

    if (f.fail()) return 1;

    std::vector<uint8_t> buf{std::istreambuf_iterator<char>(f), {}};

    /***********************************************************/
    /* 1st Experiment: 8-bit, whole data, basic Huffman        */
    /***********************************************************/
    whole_data_experiment(buf, 8);

    /***********************************************************/
    /* 2nd Experiment: 32-bit, whole data, basic Huffman       */
    /***********************************************************/
    whole_data_experiment(buf, 32);

    /***********************************************************/
    /* 3rd Experiment: 8-bit, 40MB, basic Huffman              */
    /***********************************************************/
    n_bit_experiment(buf, 8, 40);

    /***********************************************************/
    /* 4th Experiment: 32-bit, 40MB, basic Huffman             */
    /***********************************************************/
    n_bit_experiment(buf, 32, 40);

    /***********************************************************/
    /* 5th Experiment: Speed test of basic Huffman             */
    /***********************************************************/
    speed_test(buf);

    /***********************************************************/
    /* 6th Experiment: 1~127 bit, whole data, basic Huffman    */
    /***********************************************************/
    width_experiment(buf);

    /***********************************************************/
    /* 7th Experiment: NTY Coding Test                         */
    /***********************************************************/
    adaptive_huffman_textbook_NTY_code_test();

    /***********************************************************/
    /* 8th Experiment: Adaptive Huffman Encoding Test          */
    /***********************************************************/
    adaptive_huffman_textbook_example();

    /***********************************************************/
    /* 9th Experiment: 8-bit, whole data, adaptive Huffman     */
    /***********************************************************/
    adaptive_huffman_whole_data_experiment(buf, 8);

    /***********************************************************/
    /* 10th Experiment: 32-bit, whole data, adaptive Huffman   */
    /***********************************************************/
    adaptive_huffman_whole_data_experiment(buf, 32);

    /***********************************************************/
    /* 11th Experiment: 8-bit, 40MB, adaptive Huffman          */
    /***********************************************************/
    adaptive_huffman_n_bit_experiment(buf, 8, 40);

    /***********************************************************/
    /* 12th Experiment: 32-bit, 40MB, adaptive Huffman         */
    /***********************************************************/
    adaptive_huffman_n_bit_experiment(buf, 32, 40);

    /***********************************************************/
    /* 13th Experiment: Speed test of adaptive Huffman         */
    /***********************************************************/
    adaptive_huffman_speed_test(buf);

    /***********************************************************/
    /* 14th Experiment: 1~20 bit, adaptive Huffman            */
    /***********************************************************/
    adaptive_huffman_width_experiment(buf);

    /***********************************************************/
    /* 15th Experiment: 8, 16, and 32 extended Huffman         */
    /***********************************************************/
    extended_huffman(buf);
}
