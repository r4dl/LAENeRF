#include <torch/extension.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

#include <fstream>

std::vector<uint8_t> edit_grid() {
    std::vector<uint8_t> egrid;
    std::ifstream ifs("edit_grid.vec", std::ios::in | std::ifstream::binary);
    std::istream_iterator<uint8_t> iter{ifs};
    std::istream_iterator<uint8_t> end{};
    std::copy(iter, end, std::back_inserter(egrid));

    return egrid;
}

std::vector<float> load_float_vec(const char* path, int n) {
    int length = n;
    std::vector<float> pts(length, 0.f);
    std::ifstream in(path, std::ios::in | std::ios::binary);
    in.read((char *) pts.data(), length * sizeof(float));

    // see how many bytes have been read
    printf("%d\n", in.gcount() / sizeof(float));

    in.close();

    return pts;
}