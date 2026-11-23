// src/poly/hip_lt.hpp - tuned hipBLASLt solutions for the HIP build's prompt products (mmq_own.cu's BLAS path).
//
// In a HIP build every quantized prompt product is a dequantized matrix of halves times the tokens' halves, FP32
// sums, by hipblasGemmEx (mmq_own.cu).  rocBLAS's own pick for those shapes is far from the best kernel the card has:
// on an RX 7800 XT (gfx1101, ROCm 7.0.2) Inkling 2 Mini's expert products (2304 x 2304 and 2304 x 1152, 32 to 512
// tokens) ran at 2.4 to 8 TFLOPS against 8 to 39 for the best hipBLASLt solution, and its 2,048-token attention
// projections at 7 to 8 against 41 to 50 (tune_hipblaslt).  With STRATA_HIPBLASLT_TUNING naming a table that
// tools/hip/tune_hipblaslt wrote for the card and its hipBLASLt (the format of src/prefill/hipblaslt_tuning.hpp, which
// `strata` reads too), a product whose shape the table has goes to the table's solution; any other goes to
// hipblasGemmEx as before.  Unset, nothing changes.  The solutions' sums are FP32 like hipBLAS's, but in another order:
// the products are not bit for bit the same.
#pragma once

#include <cstdint>

namespace strata::poly::hip_lt {

// Y (n x t, column-major, ldy) = W^T X: W is k x n halves (a row-major n x k matrix, ld k), X is k x t halves (ld k),
// the sums FP32.  False when no table is loaded, the table has no row for (n, k, ldy), or the solution does not take
// this call: the caller makes the product itself.  `stream` is the hipStream_t.
bool gemm(const void* w, const void* x, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, void* stream);

}  // namespace strata::poly::hip_lt
