// src/poly/hip_lt.cpp - hip_lt.hpp: STRATA_HIPBLASLT_TUNING's solutions for mmq_own.cu's BLAS path (HIP builds with
// hipBLASLt).  The table, its header checks and its nearest-token-count rule are src/prefill/hipblaslt_tuning.hpp's,
// so one table format and one tuner (tools/hip/tune_hipblaslt) serve both engines.  No workspace: a solution that
// asks for one is not taken, so the expert cache keeps all of the card.
#include "hip_lt.hpp"

#include "../prefill/hipblaslt_tuning.hpp"

#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace strata::poly::hip_lt {
namespace {

using strata::prefill::hipblaslt::InputType;
using strata::prefill::hipblaslt::TuningTable;

// one call's shape: its descriptors and solution, made once
struct Call {
    bool ok = false;
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    int solution = -1;
};

struct State {
    hipblasLtHandle_t handle = nullptr;
    TuningTable table;
    std::map<std::tuple<int, int, int, int>, Call> calls;   // t, n, k, ldy
    unsigned long long launches = 0, fallbacks = 0;
    bool verbose = false;

    ~State() {
        if (verbose)
            std::fprintf(stderr, "strata-poly prompt product: hipBLASLt %llu launches, %llu by hipBLAS\n", launches,
                         fallbacks);
        for (auto& [key, call] : calls) {
            if (call.op) hipblasLtMatmulDescDestroy(call.op);
            if (call.a) hipblasLtMatrixLayoutDestroy(call.a);
            if (call.b) hipblasLtMatrixLayoutDestroy(call.b);
            if (call.c) hipblasLtMatrixLayoutDestroy(call.c);
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

std::mutex mu;

// the process's table, read once: nullptr when STRATA_HIPBLASLT_TUNING is unset or its table cannot be used
State* state() {
    static std::unique_ptr<State> s = []() -> std::unique_ptr<State> {
        const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
        if (!path || !*path) return nullptr;
        auto st = std::make_unique<State>();
        st->verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
        int version = 0, device = 0;
        hipDeviceProp_t prop{};
        if (hipblasLtCreate(&st->handle) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtGetVersion(st->handle, &version) != HIPBLAS_STATUS_SUCCESS || hipGetDevice(&device) != hipSuccess ||
            hipGetDeviceProperties(&prop, device) != hipSuccess) {
            std::fprintf(stderr, "strata-poly prompt product: hipBLASLt did not start; the products stay on hipBLAS\n");
            return nullptr;
        }
        std::string arch(prop.gcnArchName);
        if (const auto colon = arch.find(':'); colon != std::string::npos) arch.resize(colon);
        std::string err;
        if (!st->table.load(path, arch, version, err)) {
            std::fprintf(stderr, "strata-poly prompt product: %s: %s; the products stay on hipBLAS\n", path, err.c_str());
            return nullptr;
        }
        std::fprintf(stderr, "strata-poly prompt product: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                     st->table.rows().size(), arch.c_str(), version);
        return st;
    }();
    return s.get();
}

// the call's descriptors and the table's solution for it, checked against the actual shape
Call make(State& st, int t, int n, int k, int ldy) {
    Call c;
    const auto* row = st.table.closest(InputType::f16, n, k, ldy, t);
    if (!row) return c;
    const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
    if (hipblasLtMatmulDescCreate(&c.op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(c.op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(c.op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&c.a, HIP_R_16F, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&c.b, HIP_R_16F, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&c.c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS)
        return c;   // the destructor frees what was made (the entry is kept)
    std::vector<int> ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> found;
    if (hipblaslt_ext::getAlgosFromIndex(st.handle, ids, found) != HIPBLAS_STATUS_SUCCESS || found.empty() ||
        found.front().state != HIPBLAS_STATUS_SUCCESS || hipblaslt_ext::getIndexFromAlgo(found.front().algo) != row->solution_id)
        return c;
    const float alpha = 1.f, beta = 0.f;
    size_t workspace = 0;
    c.algo = found.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(st.handle, c.op, &alpha, c.a, c.b, &beta, c.c, c.c, c.algo, workspace) !=
            HIPBLAS_STATUS_SUCCESS ||
        workspace != 0)
        return c;
    c.ok = true;
    c.solution = row->solution_id;
    return c;
}

}  // namespace

bool gemm(const void* w, const void* x, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, void* stream) {
    State* st = state();
    if (!st || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n)
        return false;
    std::lock_guard<std::mutex> lock(mu);
    const auto key = std::make_tuple((int) t, (int) n, (int) k, (int) ldy);
    auto it = st->calls.find(key);
    if (it == st->calls.end()) {
        it = st->calls.emplace(key, make(*st, (int) t, (int) n, (int) k, (int) ldy)).first;
        if (st->verbose)
            std::fprintf(stderr, "strata-poly prompt product: T=%d N=%d K=%d ldy=%d -> %s %d\n", (int) t, (int) n,
                         (int) k, (int) ldy, it->second.ok ? "solution" : "hipBLAS", it->second.solution);
    }
    Call& c = it->second;
    if (!c.ok) {
        ++st->fallbacks;
        return false;
    }
    const float alpha = 1.f, beta = 0.f;
    if (hipblasLtMatmul(st->handle, c.op, &alpha, w, c.a, x, c.b, &beta, y, c.c, y, c.c, &c.algo, nullptr, 0,
                        (hipStream_t) stream) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "strata-poly prompt product: hipBLASLt solution %d failed for T=%d N=%d K=%d; hipBLAS from now on\n",
                     c.solution, (int) t, (int) n, (int) k);
        c.ok = false;   // beta is 0: hipBLAS writes the whole product again
        ++st->fallbacks;
        return false;
    }
    ++st->launches;
    return true;
}

}  // namespace strata::poly::hip_lt
