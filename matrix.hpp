#ifndef MATRIX_H
#define MATRIX_H
#include <cstddef>
#include <iostream>
#include <iomanip>

struct Matrix {
    double *mem;
    size_t rows;
    size_t cols;

    Matrix(size_t rows, size_t cols)
        : mem(new double[rows * cols]), rows(rows), cols(cols) {}
    ~Matrix() { delete[] mem; }
    double *operator[](size_t idx) { return &mem[idx * rows]; }
    double const *operator[](size_t idx) const { return &mem[idx * rows]; }
};


inline void display_matrix(std::string const &name, Matrix const &m) {
    size_t opt_size = 0;
    std::string const sep = " ";

    for (size_t i = 0; i < (m.rows * m.cols); ++i) {
        std::ostringstream oss;
        oss << m.mem[i];
        size_t size = oss.str().size();
        if (size > opt_size) {
            opt_size = size;
        }
    }

    std::cout << name << " =" << std::endl;
    for (size_t i = 0; i < m.rows; ++i) {
        std::cout << "\t";
        for (size_t j = 0; j < m.cols; ++j) {
            std::cout << std::setw(opt_size) << m[i][j] << sep;
        }
        std::cout << std::endl;
    }
}

#endif
