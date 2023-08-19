# Readme of Huffman Coding Homework, Information Theory, Spring 2023

| Chung-Yi Chen, 311551070

`data.txt` is the text file storing the experiments result.

## Usage

If the compiler supports OpenMP (GCC and Clang do), run the following command can obtain the result in the text file `out.txt`.

```
make clean all && time ./huff >out.txt
```

If `Python`, `Numpy`, and `Matplotlib` are installed, use the following command instead.

```
make clean plot && time ./huff >out.txt
```

Remember to modify the path of Python header, Numpy include directory, and libpython location.
