#ifndef WARPCORE_BASE_CUH
#define WARPCORE_BASE_CUH

#include <algorithm>
#include <assert.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <helpers/cuda_helpers.cuh>
#include <helpers/packed_types.cuh>
#include <limits>

#include "primes.hpp"

namespace warpcore {

namespace cg = cooperative_groups;

using index_t       = std::uint64_t;
using status_base_t = std::uint32_t;

namespace detail {

HOSTQUALIFIER INLINEQUALIFIER index_t get_valid_capacity(index_t min_capacity,
                                                         index_t cg_size) noexcept
{
  const auto x = SDIV(min_capacity, cg_size);
  const auto y = std::lower_bound(primes.begin(), primes.end(), x);
  return (y == primes.end()) ? 0 : (*y) * cg_size;
}

}  // namespace detail

}  // namespace warpcore

// TODO move to defaults and expose as constexpr
#ifdef __CUDACC_DEBUG__
#ifndef WARPCORE_BLOCKSIZE
#define WARPCORE_BLOCKSIZE 128
#endif
#else
#ifndef WARPCORE_BLOCKSIZE
#define WARPCORE_BLOCKSIZE MAXBLOCKSIZE  // MAXBLOCKSIZE defined in cuda_helpers
#endif
#endif

#include "checks.cuh"
#include "defaults.cuh"
#include "gpu_engine.cuh"
#include "hashers.cuh"
#include "probing_schemes.cuh"
#include "status.cuh"
#include "storage.cuh"
#include "tags.cuh"

#endif /* WARPCORE_BASE_CUH */
