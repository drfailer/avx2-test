#include <cstring>
#include <cassert>
#include <immintrin.h>
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


double mm512_haddreduce_d(__m512d v) {
    __m256d vlow  = _mm512_castpd512_pd256(v);   // low 256
    __m256d vhigh = _mm512_extractf64x4_pd(v, 1); // high 256
            vlow  = _mm256_add_pd(vlow, vhigh);     // reduce down to 256
    return  mm256_haddreduce_d(vlow);
}

void matmul_avx256(Matrix const &A, Matrix const &B, Matrix &C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            __m256d sum256 = _mm256_set_pd(0, 0, 0, 0);
            double sum = 0;
            size_t block_size = 256 / (sizeof(double) * 8);
            size_t N = A.cols / block_size;

            for (size_t k = 0; k < N; ++k) {
                size_t kk = k * block_size;
                __m256d a = _mm256_load_pd(&A[i][kk]);
                __m256d b = _mm256_setr_pd(B[kk][j], B[kk + 1][j], B[kk + 2][j],
                                          B[kk + 3][j]);
                __m256d mul = _mm256_mul_pd(a, b);
                sum256 = _mm256_add_pd(sum256, mul);
            }

            for (size_t k = A.cols - A.cols%block_size; k < A.cols; ++k) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = mm256_haddreduce_d(sum256) + sum;
        }
    }
}

void matmul_avx512(Matrix const &A, Matrix const &B, Matrix &C) {
    assert(A.cols == B.rows);
    assert(C.rows == A.rows && C.cols == B.cols);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t j = 0; j < B.cols; ++j) {
            __m512d sum512 = _mm512_set_pd(0, 0, 0, 0, 0, 0, 0, 0);
            double sum = 0;
            size_t block_size = 512 / (sizeof(double) * 8);
            size_t N = A.cols / block_size;

            for (size_t k = 0; k < N; ++k) {
                size_t kk = k * block_size;
                __m512d a = _mm512_load_pd(&A[i][kk]);
                __m512d b = _mm512_setr_pd(
                    B[kk][j], B[kk + 1][j], B[kk + 2][j], B[kk + 3][j],
                    B[kk][j + 4], B[kk + 5][j], B[kk + 6][j], B[kk + 7][j]);
                __m512d mul = _mm512_mul_pd(a, b);
                sum512 = _mm512_add_pd(sum512, mul);
            }

            for (size_t k = A.cols - A.cols%block_size; k < A.cols; ++k) {
                sum += A[i][k] * B[k][j];
            }

            C[i][j] = mm512_haddreduce_d(sum512) + sum;
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

int assert_equal(Matrix const &C1, Matrix const &C2) {
    assert(C1.rows == C2.rows && C1.cols == C2.cols);
    for (size_t i = 0; i < C1.rows * C1.cols; ++i) {
        if (C1.mem[i] != C2.mem[i]) {
              std::cout << "error[" << i << "]:" << C1.mem[i] << " != "
                        << C2.mem[i] << std::endl;
              return 1;
        }
    }
    std::cout << "matrix are equals" << std::endl;
    return 0;
}

int main(int, char**) {
    constexpr size_t m = 1000;
    constexpr size_t n = 1000;
    constexpr size_t k = 1000;
    Matrix A(m, n);
    Matrix B(n, k);
    Matrix C1(m, k);
    Matrix C2(m, k);
    Matrix C3(m, k);

    init(A, B, C1);
    timer_start(matmul);
    matmul(A, B, C1);
    timer_end(matmul);

    if (m < 8 && k < 10) display_matrix("C1", C1);

    timer_start(matmul_avx2);
    matmul_avx256(A, B, C2);
    timer_end(matmul_avx2);


    timer_start(matmul_avx512);
    matmul_avx256(A, B, C3);
    timer_end(matmul_avx512);


    if (m < 8 && k < 10) display_matrix("C2", C2);

    timer_report_prec(matmul, milliseconds);
    timer_report_prec(matmul_avx2, milliseconds);
    timer_report_prec(matmul_avx512, milliseconds);

    assert_equal(C1, C2);
    assert_equal(C1, C3);
    return 0;
}
