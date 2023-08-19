# Comparison of Compression Efficiency of Huffman Coding and Arithmetic Coding

## Warning

If you are taking the course "Information Theory" or "Information Theory and Data Compression Practices" taught by Prof. Chun-Jen Tsai, you should not copy the code and submit it as your own assignment. My motivation is to provide a guide for students who do not know how to start the assignments.

## Introduction

This repo only implements the **encoder** part of each algorithm for evaluating the compression efficiency of different compression algorithms. The implemented algorithms are Huffman coding, adaptive Huffman coding, extended Huffman coding, arithmetic coding, and arithmetic coding with context-based compression algorithm prediction by partial matching (PPM, including ppma, ppmb, and ppmc). These algorithms support different template arguments, making experiments easier.

## Usage

You should first change the `./alexnet.pth` of `std::fstream f{"./alexnet.pth", std::ios::in|std::ios::binary}` to your filename for compression in the main function. Then, follow the readme and you can reproduce the result.

To obtain more data from different settings, I have done several optimizations including thread-level parallelism and data structure modification, which make the implementation not fits the pseudocode as you expect. I highly recommend reading my **reports** to see my implementation decision.

## Note

By passing the template argument `show_step` to `true`, you can dump the steps of arithmetic coding. This may be helpful if you want to figure out the behavior of arithmetic coding with PPM. You can check `out.txt` to see what the output looks like.
