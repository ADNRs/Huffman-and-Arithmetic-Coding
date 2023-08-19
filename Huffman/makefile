CXX       := g++
CXXFLAGS  := -fopenmp -std=c++20 -O3 -Wall -Wextra -Werror -DNDEBUG
NUMPY     := -I${HOME}/.local/lib/python3.10/site-packages/numpy/core/include/
PYTHON    := -I/usr/include/python3.10
LIBPYTHON := -lpython3.10
TARGET    := huff
SRC       := main.cpp

default: clean all

.PHONY: all
all: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)
	rm -rf images

.PHONY: plot
plot: $(SRC)
	mkdir images
	$(CXX) $(CXXFLAGS) -DPLOT $(PYTHON) $(NUMPY) $^ -o $(TARGET) $(LIBPYTHON)
