#include <cstring>
#include <cassert>
#include "matrix.hpp"
#include "immintrin.h"
#include "timer.h"

// C = A * B
void matmul(Matrix const &A, Matrix const &B, Matrix &C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            double sum = 0;
            for (size_t k = 0; k < A.cols; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// https://stackoverflow.com/questions/49941645/get-sum-of-values-stored-in-m256d-with-sse-avx
double mm256_haddreduce_d(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);   // low 128
    __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
            vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow); // high 64
    return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to 64
}

void matmul_avx2(Matrix const &A, Matrix const &B, Matrix &C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            __m256d sum256 = _mm256_set_pd(0, 0, 0, 0);
            double sum = 0;
            size_t block_size = 256 / sizeof(double);
            size_t N = A.cols / block_size;

            for (size_t k = 0; k < N; ++k) {
                __m256d a = _mm256_load_pd(&A[i][k*4]);
                __m256d b = _mm256_set_pd(B[k * 4][i], B[k * 4 + 1][i],
                                          B[k * 4 + 2][i], B[k * 4 + 3][i]);
                __m256d mul = _mm256_mul_pd(a, b);
                _mm256_add_pd(sum256, mul);
            }

            for (size_t k = A.cols - A.cols%block_size; k < A.cols; ++k) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = mm256_haddreduce_d(sum256) + sum;
        }
    }
}

void init(Matrix &A, Matrix &B, Matrix &C) {
    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < A.cols; ++j) {
            A[i][j] = i + j;
        }
    }

    for (size_t i = 0; i < B.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            B[i][j] = i * j;
        }
    }

    memset(C.mem, 0, C.rows * C.cols * sizeof(*C.mem));
}

int main(int, char**) {
    constexpr size_t m = 1000;
    constexpr size_t n = 1000;
    constexpr size_t k = 1000;
    Matrix A(m, n);
    Matrix B(n, k);
    Matrix C(m, k);

    init(A, B, C);
    timer_start(matmul);
    matmul(A, B, C);
    timer_end(matmul);

    timer_start(matmul_avx2);
    matmul_avx2(A, B, C);
    timer_end(matmul_avx2);

    timer_report_prec(matmul, milliseconds);
    timer_report_prec(matmul_avx2, milliseconds);
    return 0;
}
