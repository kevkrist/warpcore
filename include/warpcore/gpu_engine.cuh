#ifndef WARPCORE_GPU_ENGINE_CUH
#define WARPCORE_GPU_ENGINE_CUH

#include <helpers/cuda_helpers.cuh>

/*! \brief CUDA kernels
 */
namespace warpcore::kernels {

template <class T, T Val = 0>
GLOBALQUALIFIER void memset(T* const arr, const index_t num)
{
  const index_t tid = helpers::global_thread_id();

  if (tid < num) { arr[tid] = Val; }
}

template <class Func, class Core>
GLOBALQUALIFIER void for_each(Func f, const Core core)
{
  const index_t tid = helpers::global_thread_id();

  if (tid < core.capacity()) {
    auto&& pair = core.table_[tid];
    if (core.is_valid_key(pair.key)) { f(pair.key, pair.value); }
  }
}

template <class Func, class Core>
GLOBALQUALIFIER void for_each_unique_key(Func f, const Core core)
{
  using index_type          = typename Core::index_type;
  using probing_scheme_type = typename Core::probing_scheme_type;

  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < core.capacity()) {
    // for valid entry in table check if this entry is the first of its key
    auto search_key = core.table_[gid].key;
    if (core.is_valid_key(search_key)) {
      probing_scheme_type iter(core.capacity(), core.capacity(), group);

      for (index_type i = iter.begin(search_key, core.seed_); i != iter.end(); i = iter.next()) {
        const auto table_key = core.table_[i].key;
        const auto hit       = (table_key == search_key);
        const auto hit_mask  = group.ballot(hit);

        const auto leader = ffs(hit_mask) - 1;

        // check if search_key is the first entry for this key
        if (group.thread_rank() == leader && i == gid) { f(table_key); }

        if (group.any(hit)) { return; }
      }
    }
  }
}

template <class Func, class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void for_each(Func f,
                              const typename Core::key_type* const keys_in,
                              const index_t num_in,
                              const Core core,
                              const index_t probing_length = defaults::probing_length(),
                              typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    index_t num_values;

    const auto status = core.for_each(f, keys_in[gid], num_values, group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

//--------------------------------------------------------------------------------------------------
// for Core = BloomFilter (dedicated namespace)
//--------------------------------------------------------------------------------------------------
namespace bloom_filter {

template <class Core>
GLOBALQUALIFIER void insert(const typename Core::key_type* const keys_in,
                            const index_t num_in,
                            Core core)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) { core.insert(keys_in[gid], group); }
}

template <class Core, typename Filter, typename FilterValueType>
GLOBALQUALIFIER void insert_if(const typename Core::key_type* const keys_in,
                               Filter f,
                               const FilterValueType* const values_in,
                               const index_t num_in,
                               Core core)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in && f(values_in[gid])) { core.insert(keys_in[gid], group); }
}

template <class Core>
GLOBALQUALIFIER void retrieve(const typename Core::key_type* const keys_in,
                              const index_t num_in,
                              typename Core::value_type* const values_out,
                              const Core core)
{
  const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    typename Core::value_type value = core.retrieve(keys_in[gid], group);

    if (group.thread_rank() == 0) { values_out[gid] = value; }
  }
}

template <class Core, typename Writer>
GLOBALQUALIFIER void retrieve_write(const typename Core::key_type* const keys_in,
                                    const index_t num_in,
                                    int* counter,
                                    Writer writer,
                                    const Core core)
{
  const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto key = keys_in[gid];
    if (core.retrieve(key, group)) {
      if (group.thread_rank() == 0) {
        const auto write_index = helpers::atomicAggInc(counter);
        writer(write_index, gid);
      }
    }
  }
}

template <class Core, typename Filter, typename FilterTypeValue, typename Writer>
GLOBALQUALIFIER void retrieve_write_if(const typename Core::key_type* const keys_in,
                                       Filter f,
                                       const FilterTypeValue* const filter_values,
                                       const index_t num_in,
                                       int* counter,
                                       Writer writer,
                                       const Core core)
{
  const index_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto filter_value = filter_values[gid];
    if (f(filter_value)) {
      const auto key = keys_in[gid];
      if (core.retrieve(key, group)) {
        if (group.thread_rank() == 0) {
          const auto write_index = helpers::atomicAggInc(counter);
          writer(write_index, gid, key, filter_value);
        }
      }
    }
  }
}

}  // namespace bloom_filter

//--------------------------------------------------------------------------------------------------
// for Core = HashSet
//--------------------------------------------------------------------------------------------------
template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void insert(const typename Core::key_type* const keys_in,
                            const index_t num_in,
                            Core core,
                            const index_t probing_length = defaults::probing_length(),
                            typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto status = core.insert(keys_in[gid], group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

template <class Core,
          typename Filter,
          typename FilterValueType,
          class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void insert_if(const typename Core::key_type* const keys_in,
                               Filter f,
                               const FilterValueType* const filter_values_in,
                               const index_t num_in,
                               Core core,
                               const index_t probing_length = defaults::probing_length(),
                               typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in && f(filter_values_in[gid])) {
    const auto status = core.insert(keys_in[gid], group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

namespace hash_set {

template <class Core, typename Writer, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve_write(const typename Core::key_type* const keys_in,
                                    const index_t num_in,
                                    int* counter,
                                    Writer writer,
                                    const Core core,
                                    const index_t probing_length = defaults::probing_length(),
                                    typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto key = keys_in[gid];
    bool found;
    const auto status = core.retrieve(key, found, group, probing_length);

    if (found) {
      if (group.thread_rank() == 0) {
        auto write_index = helpers::atomicAggInc(counter);
        writer(write_index, gid, key);
      }

      StatusHandler::handle(status, status_out, gid);
    }
  }
}

template <class Core,
          typename Filter,
          typename FilterValueType,
          typename Writer,
          class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve_write_if(
  const typename Core::key_type* const keys_in,
  Filter f,
  const FilterValueType* const filter_values_in,
  const index_t num_in,
  int* counter,
  Writer writer,
  const Core core,
  const index_t probing_length                        = defaults::probing_length(),
  typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto filter_value = filter_values_in[gid];

    if (f(filter_value)) {
      const auto key = keys_in[gid];
      bool found;
      const auto status = core.retrieve(key, found, group, probing_length);

      if (found) {
        if (group.thread_rank() == 0) {
          auto write_index = helpers::atomicAggInc(counter);
          writer(write_index, gid, key, filter_value);
        }

        StatusHandler::handle(status, status_out, gid);
      }
    }
  }
}

}  // namespace hash_set

//--------------------------------------------------------------------------------------------------
// for Core = SingleValueHashTable
//--------------------------------------------------------------------------------------------------
template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void insert(const typename Core::key_type* const keys_in,
                            const typename Core::value_type* const values_in,
                            const index_t num_in,
                            Core core,
                            const index_t probing_length = defaults::probing_length(),
                            typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto status = core.insert(keys_in[gid], values_in[gid], group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

// apply filter to separate array of filter values
template <class Core,
          typename Filter,
          typename FilterValueType,
          class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void insert_if(const typename Core::key_type* const keys_in,
                               const typename Core::value_type* const values_in,
                               Filter f,
                               const FilterValueType* const filter_values,
                               const index_t num_in,
                               Core core,
                               const index_t probing_length = defaults::probing_length(),
                               typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in && f(filter_values[gid])) {
    const auto status = core.insert(keys_in[gid], values_in[gid], group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

// apply filter to values to insert into hash table
template <class Core, typename Filter, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void insert_if(const typename Core::key_type* const keys_in,
                               const typename Core::value_type* const values_in,
                               Filter f,
                               const index_t num_in,
                               Core core,
                               const index_t probing_length = defaults::probing_length(),
                               typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto value = values_in[gid];
    if (f(value)) {
      const auto status = core.insert(keys_in[gid], value, group, probing_length);

      if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
    }
  }
}

template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve(const typename Core::key_type* const keys_in,
                              const index_t num_in,
                              typename Core::value_type* const values_out,
                              const Core core,
                              const index_t probing_length = defaults::probing_length(),
                              typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    typename Core::value_type value_out;
    const auto status = core.retrieve(keys_in[gid], value_out, group, probing_length);

    if (group.thread_rank() == 0) {
      if (!status.has_any()) { values_out[gid] = value_out; }
      StatusHandler::handle(status, status_out, gid);
    }
  }
}

template <class Core, typename Writer, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve_write(const typename Core::key_type* const keys_in,
                                    const index_t num_in,
                                    int* counter,
                                    Writer writer,
                                    const Core core,
                                    const index_t probing_length = defaults::probing_length(),
                                    typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    typename Core::value_type value_out;

    const auto key    = keys_in[gid];
    const auto status = core.retrieve(key, value_out, group, probing_length);

    if (group.thread_rank() == 0) {
      if (!status.has_any()) {
        auto write_index = helpers::atomicAggInc(counter);
        writer(write_index, gid, key, value_out);
      }

      StatusHandler::handle(status, status_out, gid);
    }
  }
}

template <class Core,
          typename Filter,
          typename FilterValueType,
          typename Writer,
          class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve_write_if(
  const typename Core::key_type* const keys_in,
  Filter f,
  const FilterValueType* const filter_values,
  const index_t num_in,
  int* counter,
  Writer writer,
  const Core core,
  const index_t probing_length                        = defaults::probing_length(),
  typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto filter_value = filter_values[gid];

    if (f(filter_value)) {
      typename Core::value_type value_out;

      const auto key    = keys_in[gid];
      const auto status = core.retrieve(key, value_out, group, probing_length);

      if (group.thread_rank() == 0) {
        if (!status.has_any()) {
          auto write_index = helpers::atomicAggInc(counter);
          writer(write_index, gid, key, value_out, filter_value);
        }

        StatusHandler::handle(status, status_out, gid);
      }
    }
  }
}

template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void retrieve(const typename Core::key_type* const keys_in,
                              const index_t num_in,
                              const index_t* const begin_offsets_in,
                              const index_t* const end_offsets_in,
                              typename Core::value_type* const values_out,
                              const Core core,
                              const index_t probing_length = defaults::probing_length(),
                              typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  using status_type = typename Core::status_type;

  if (gid < num_in) {
    index_t num_out;

    auto status = core.retrieve(
      keys_in[gid], values_out + begin_offsets_in[gid], num_out, group, probing_length);

    if (group.thread_rank() == 0) {
      const auto num_prev = end_offsets_in[gid] - begin_offsets_in[gid];

      if (num_prev != num_out) {
        // printf("%llu %llu\n", num_prev, num_out);
        core.device_join_status(status_type::invalid_phase_overlap());
        status += status_type::invalid_phase_overlap();
      }

      StatusHandler::handle(status, status_out, gid);
    }
  }
}

template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void erase(const typename Core::key_type* const keys_in,
                           const index_t num_in,
                           Core core,
                           const index_t probing_length = defaults::probing_length(),
                           typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    const auto status = core.erase(keys_in[gid], group, probing_length);

    if (group.thread_rank() == 0) { StatusHandler::handle(status, status_out, gid); }
  }
}

template <class Core>
GLOBALQUALIFIER void size(index_t* const num_out, const Core core)
{
  __shared__ index_t smem;

  const index_t tid = helpers::global_thread_id();
  const auto block  = cg::this_thread_block();

  if (tid < core.capacity()) {
    const bool empty = !core.is_valid_key(core.table_[tid].key);

    if (block.thread_rank() == 0) { smem = 0; }

    block.sync();

    if (!empty) {
      const auto active_threads = cg::coalesced_threads();

      if (active_threads.thread_rank() == 0) { atomicAdd(&smem, active_threads.size()); }
    }

    block.sync();

    if (block.thread_rank() == 0 && smem != 0) { atomicAdd(num_out, smem); }
  }
}

// for Core = MultiBucketHashTable
template <class Core>
GLOBALQUALIFIER void num_values(index_t* const num_out, const Core core)
{
  __shared__ index_t smem;

  const index_t tid = helpers::global_thread_id();
  const auto block  = cg::this_thread_block();

  if (tid < core.capacity()) {
    const bool empty = !core.is_valid_key(core.table_[tid].key);

    if (block.thread_rank() == 0) { smem = 0; }

    block.sync();

    index_t value_count = 0;
    if (!empty) {
      const auto bucket = core.table_[tid].value;
#pragma unroll
      for (int b = 0; b < core.bucket_size(); ++b) {
        const auto& value = bucket[b];
        if (value != core.empty_value()) ++value_count;
      }

      // TODO warp reduce
      atomicAdd(&smem, value_count);
    }

    block.sync();

    if (block.thread_rank() == 0 && smem != 0) { atomicAdd(num_out, smem); }
  }
}

template <class Core, class StatusHandler = defaults::status_handler_t>
GLOBALQUALIFIER void num_values(const typename Core::key_type* const keys_in,
                                const index_t num_in,
                                index_t* const num_out,
                                index_t* const num_per_key_out,
                                const Core core,
                                const index_t probing_length = defaults::probing_length(),
                                typename StatusHandler::base_type* const status_out = nullptr)
{
  const index_t tid = helpers::global_thread_id();
  const index_t gid = tid / Core::cg_size();
  const auto group  = cg::tiled_partition<Core::cg_size()>(cg::this_thread_block());

  if (gid < num_in) {
    index_t num = 0;

    const auto status = core.num_values(keys_in[gid], num, group, probing_length);

    if (group.thread_rank() == 0) {
      if (num_per_key_out != nullptr) { num_per_key_out[gid] = num; }

      if (num != 0) { atomicAdd(num_out, num); }

      StatusHandler::handle(status, status_out, gid);
    }
  }
}

}  // namespace warpcore::kernels

#endif /* WARPCORE_GPU_ENGINE_CUH */