#ifndef WARPCORE_SINGLE_VALUE_HASH_TABLE_CUH
#define WARPCORE_SINGLE_VALUE_HASH_TABLE_CUH

#include "base.cuh"
#include "defaults.cuh"

namespace warpcore {

// forward declaration of friends
template <class Key,
          class Value,
          Key EmptyKey,
          Key TombstoneKey,
          class ProbingScheme,
          class TableStorage,
          index_t TempMemoryBytes>
class CountingHashTable;

/*! \brief single-value hash table
 * \tparam Key key type (\c std::uint32_t or \c std::uint64_t)
 * \tparam Value value type
 * \tparam EmptyKey key which represents an empty slot
 * \tparam TombstoneKey key which represents an erased slot
 * \tparam ProbingScheme probing scheme from \c warpcore::probing_schemes
 * \tparam TableStorage memory layout from \c warpcore::storage::key_value
 * \tparam TempMemoryBytes size of temporary storage (typically a few kB)
 */
template <class Key,
          class Value,
          Key EmptyKey            = defaults::empty_key<Key>(),
          Key TombstoneKey        = defaults::tombstone_key<Key>(),
          class ProbingScheme     = defaults::probing_scheme_t<Key, 8>,
          class TableStorage      = defaults::table_storage_t<Key, Value>,
          index_t TempMemoryBytes = defaults::temp_memory_bytes()>
class SingleValueHashTable {
  static_assert(checks::is_valid_key_type<Key>(), "invalid key type");

  static_assert(EmptyKey != TombstoneKey, "empty key and tombstone key must not be identical");

  static_assert(checks::is_probing_scheme<ProbingScheme>(), "not a valid probing scheme type");

  static_assert(std::is_same<typename ProbingScheme::key_type, Key>::value,
                "probing key type differs from table's key type");

  static_assert(checks::is_key_value_storage<TableStorage>(), "not a valid storage type");

  static_assert(std::is_same<typename TableStorage::key_type, Key>::value,
                "storage's key type differs from table's key type");

  static_assert(std::is_same<typename TableStorage::value_type, Value>::value,
                "storage's value type differs from table's value type");

  static_assert(TempMemoryBytes >= sizeof(index_t),
                "temporary storage must at least be of size index_type");

  using temp_type = storage::CyclicStore<index_t>;

 public:
  using key_type    = Key;
  using value_type  = Value;
  using index_type  = index_t;
  using status_type = Status;

  /*! \brief get empty key
   * \return empty key
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr key_type empty_key() noexcept
  {
    return EmptyKey;
  }

  /*! \brief get tombstone key
   * \return tombstone key
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr key_type tombstone_key() noexcept
  {
    return TombstoneKey;
  }

  /*! \brief get cooperative group size
   * \return cooperative group size
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr index_type cg_size() noexcept
  {
    return ProbingScheme::cg_size();
  }

  /*! \brief constructor
   * \param[in] min_capacity minimum number of slots in the hash table
   * \param[in] seed random seed
   * \param[in] no_init whether to initialize the table at construction or not
   */
  HOSTQUALIFIER INLINEQUALIFIER explicit SingleValueHashTable(
    const index_type min_capacity,
    const key_type seed = defaults::seed<key_type>(),
    const bool no_init  = false) noexcept
    : status_(nullptr),
      table_(detail::get_valid_capacity(min_capacity, cg_size())),
      temp_(TempMemoryBytes / sizeof(index_type)),
      seed_(seed),
      is_initialized_(false),
      is_copy_(false)
  {
    cudaMalloc(&status_, sizeof(status_type));

    assign_status(table_.status() + temp_.status());

    if (!no_init) init();
  }

  /*! \brief copy-constructor (shallow)
   *  \param[in] object to be copied
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER SingleValueHashTable(const SingleValueHashTable& o) noexcept
    : status_(o.status_),
      table_(o.table_),
      temp_(o.temp_),
      seed_(o.seed_),
      is_initialized_(o.is_initialized_),
      is_copy_(true)
  {
  }

  /*! \brief move-constructor
   *  \param[in] object to be moved
   */
  HOSTQUALIFIER INLINEQUALIFIER SingleValueHashTable(SingleValueHashTable&& o) noexcept
    : status_(std::move(o.status_)),
      table_(std::move(o.table_)),
      temp_(std::move(o.temp_)),
      seed_(std::move(o.seed_)),
      is_initialized_(std::move(o.is_initialized_)),
      is_copy_(std::move(o.is_copy_))
  {
    o.is_copy_ = true;
  }

#ifndef __CUDA_ARCH__
  /*! \brief destructor
   */
  HOSTQUALIFIER INLINEQUALIFIER ~SingleValueHashTable() noexcept
  {
    if (!is_copy_) {
      if (status_ != nullptr) cudaFree(status_);
    }
  }
#endif

  /*! \brief (re)initialize the hash table
   * \param[in] seed random seed
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void init(const key_type seed,
                                          cudaStream_t stream = cudaStreamDefault) noexcept
  {
    is_initialized_ = false;

    seed_ = seed;
    if (!table_.status().has_not_initialized() && !temp_.status().has_not_initialized()) {
      table_.init_keys(empty_key(), stream);

      assign_status(table_.status() + temp_.status(), stream);

      is_initialized_ = true;
    }
  }

  /*! \brief (re)initialize the hash table
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void init(cudaStream_t stream = cudaStreamDefault) noexcept
  {
    init(seed_, stream);
  }

  /*! \brief initialize values of the hash table (for GROUP BY)
   * \param[in] value value to initialize table values to
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void init_values(const value_type value,
                                                 cudaStream_t stream = cudaStreamDefault) noexcept
  {
    table_.init_values(value, stream);
  }

  /*! \brief inserts a key into the hash table
   * \param[in] key_in key to insert into the hash table
   * \param[in] value_in value that corresponds to \c key_in
   * \param[in] group cooperative group
   * \param[in] probing_length maximum number of probing attempts
   * \return status (per thread)
   */
  DEVICEQUALIFIER INLINEQUALIFIER status_type
  insert(const key_type key_in,
         const value_type& value_in,
         const cg::thread_block_tile<cg_size()>& group,
         const index_type probing_length = defaults::probing_length()) noexcept
  {
    status_type status = status_type::unknown_error();

    value_type* value_ptr = insert_impl(key_in, status, group, probing_length);

    if (group.thread_rank() == 0 && value_ptr != nullptr) { *value_ptr = value_in; }

    return status;
  }

  /*! \brief aggregates values for given key (values must be properly initialized)
   * \tparam Atomic the functor wrapper for the atomic operation to perform for aggregation
   * \param[in] key_in key to insert into the hash table
   * \param[in] value_in value that corresponds to \c key_in
   * \param[in] atomic_aggregator device lambda wrapping atomic function to aggregate with
   * \param[in] group cooperative group
   * \param[in] probing_length maximum number of probing attempts
   * \return status (per thread)
   */
  template <typename Atomic>
  DEVICEQUALIFIER INLINEQUALIFIER status_type
  insert(const key_type key_in,
         const value_type& value_in,
         Atomic atomic_aggregator,
         const cg::thread_block_tile<cg_size()>& group,
         const index_type probing_length = defaults::probing_length()) noexcept
  {
    status_type status = status_type::unknown_error();

    value_type* value_ptr = insert_impl(key_in, status, group, probing_length);

    if (group.thread_rank() == 0 && value_ptr != nullptr) {
      atomic_aggregator(value_ptr, value_in);
    }

    // Remove duplicate key status
    return status - warpcore::Status::duplicate_key();
  }

  /*! \brief insert a set of keys into the hash table
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to insert into the hash table
   * \param[in] values_in corresponds values to \c keys_in
   * \param[in] num_in number of keys to insert
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information per key
   */
  template <class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void insert(
    const key_type* const keys_in,
    const value_type* const values_in,
    const index_type num_in,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::insert<SingleValueHashTable, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, values_in, num_in, *this, probing_length, status_out);
  }

  /*! \brief insert a set of keys into the hash table, subject to a filter on the input values
   * \tparam Filter the filter functor to apply to values_in
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to insert into the hash table
   * \param[in] values_in corresponds values to \c keys_in
   * \param[in] f filter to apply to values_in
   * \param[in] num_in number of keys to insert
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information per key
   */
  template <typename Filter, class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void insert_if(
    const key_type* const keys_in,
    const value_type* const values_in,
    Filter f,
    const index_type num_in,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::insert_if<SingleValueHashTable, Filter, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, values_in, f, num_in, *this, probing_length, status_out);
  }

  /*! \brief insert a set of keys into the hash table, subject to a filter on external values
   * \tparam Filter the filter functor to apply to filter_values
   * \tparam FilterType the type of filter_values
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to insert into the hash table
   * \param[in] values_in corresponds values to \c keys_in
   * \param[in] f filter to apply to values_in
   * \param[in] num_in number of keys to insert
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information per key
   */
  template <typename Filter, typename FilterType, class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void insert_if(
    const key_type* const keys_in,
    const value_type* const values_in,
    Filter f,
    FilterType* filter_values,
    const index_type num_in,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::insert_if<SingleValueHashTable, Filter, FilterType, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, values_in, f, filter_values, num_in, *this, probing_length, status_out);
  }

  /*! \brief retrieves a key from the hash table
   * \param[in] key_in key to retrieve from the hash table
   * \param[out] value_out value for \c key_in
   * \param[in] group cooperative group
   * \param[in] probing_length maximum number of probing attempts
   * \return status (per thread)
   */
  DEVICEQUALIFIER INLINEQUALIFIER status_type
  retrieve(const key_type key_in,
           value_type& value_out,
           const cg::thread_block_tile<cg_size()>& group,
           const index_type probing_length = defaults::probing_length()) const noexcept
  {
    if (!is_initialized_) return status_type::not_initialized();

    if (!is_valid_key(key_in)) {
      status_->atomic_join(status_type::invalid_key());
      return status_type::invalid_key();
    }

    ProbingScheme iter(capacity(), probing_length, group);

    for (index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next()) {
      key_type table_key  = table_[i].key;
      const bool hit      = (table_key == key_in);
      const auto hit_mask = group.ballot(hit);

      if (hit_mask) {
        const auto leader = ffs(hit_mask) - 1;
        value_out         = table_[group.shfl(i, leader)].value;

        return status_type::none();
      }

      if (group.any(is_empty_key(table_key))) {
        status_->atomic_join(status_type::key_not_found());
        return status_type::key_not_found();
      }
    }

    status_->atomic_join(status_type::probing_length_exceeded());
    return status_type::probing_length_exceeded();
  }

  template <typename CounterType>
  DEVICEQUALIFIER INLINEQUALIFIER status_type
  retrieve_debug(const key_type key_in,
                 value_type& value_out,
                 CounterType& num_iter_out,
                 const cg::thread_block_tile<cg_size()>& group,
                 const index_type probing_length = defaults::probing_length()) const noexcept
  {
    if (!is_initialized_) return status_type::not_initialized();

    if (!is_valid_key(key_in)) {
      status_->atomic_join(status_type::invalid_key());
      return status_type::invalid_key();
    }

    ProbingScheme iter(capacity(), probing_length, group);

    num_iter_out = 0;
    CounterType counter  = 1;
    for (index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next()) {
      key_type table_key  = table_[i].key;
      const bool hit      = (table_key == key_in);
      const auto hit_mask = group.ballot(hit);

      if (hit_mask) {
        const auto leader = ffs(hit_mask) - 1;  // extracts leader (integral) from bit array
        value_out         = table_[group.shfl(i, leader)].value;
        num_iter_out      = counter;
        return status_type::none();
      }

      if (group.any(is_empty_key(table_key))) {
        status_->atomic_join(status_type::key_not_found());
        return status_type::key_not_found();
      }
      ++counter;
    }

    status_->atomic_join(status_type::probing_length_exceeded());
    return status_type::probing_length_exceeded();
  }

  /*! \brief retrieve a set of keys from the hash table
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to retrieve from the hash table
   * \param[in] num_in number of keys to retrieve
   * \param[out] values_out retrieved values of keys in \c key_in
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information (per key)
   */
  template <class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void retrieve(
    const key_type* const keys_in,
    const index_type num_in,
    value_type* const values_out,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) const noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::retrieve<SingleValueHashTable, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, num_in, values_out, *this, probing_length, status_out);
  }

  /*! \brief retrieve a set of keys from the hash table and write out corresponding values
   * \tparam Writer functor / device lambda for writing column values out
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to retrieve from the hash table
   * \param[in] num_in number of keys to retrieve
   * \param[out] counter counter of keys and values retrieved (pointer with reference initialized
   * to zero)
   * \param[in] writer device functor or lambda for writing out additional table columns
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information (per key)
   */
  template <typename Writer, class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void retrieve_write(
    const key_type* const keys_in,
    const index_type num_in,
    int* counter,
    Writer writer,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) const noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::retrieve_write<SingleValueHashTable, Writer, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, num_in, counter, writer, *this, probing_length, status_out);
  }

  /*! \brief retrieve a set of keys from the hash table and write out values subject to filter
   * \tparam Filter the filter functor to apply to filter_values
   * \tparam FilterValueType the type of filter_values
   * \tparam Writer the functor type for writing out additional columns
   * \tparam StatusHandler handles returned status per key (see \c status_handlers)
   * \param[in] keys_in pointer to keys to retrieve from the hash table
   * \param[in] f predicate functor instance to apply to filter values
   * \param[in] filter_values values to which to apply f
   * \param[in] num_in number of keys to retrieve
   * \param[out] counter counter of keys and values retrieved (initialize reference to zero)
   * \param[in] writer device functor instance or lambda for writing out additional columns
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] probing_length maximum number of probing attempts
   * \param[out] status_out status information (per key)
   */
  template <typename Filter,
            typename FilterValueType,
            typename Writer,
            class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void retrieve_write_if(
    const key_type* const keys_in,
    Filter f,
    const FilterValueType* const filter_values,
    const index_type num_in,
    int* counter,
    Writer writer,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) const noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::retrieve_write_if<SingleValueHashTable, Filter, FilterValueType, Writer, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, f, filter_values, num_in, counter, writer, *this, probing_length, status_out);
  }

  template <typename Filter,
            typename FilterValueType,
            typename IterCounterType,
            typename Writer,
            class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void retrieve_write_if_debug(
    const key_type* const keys_in,
    Filter f,
    const FilterValueType* const filter_values,
    const index_type num_in,
    IterCounterType* const num_iter_out,
    int* counter,
    Writer writer,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) const noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::retrieve_write_if_debug<SingleValueHashTable,
                                     Filter,
                                     FilterValueType,
                                     IterCounterType,
                                     Writer,
                                     StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in,
        f,
        filter_values,
        num_in,
        num_iter_out,
        counter,
        writer,
        *this,
        probing_length,
        status_out);
  }

  /*! \brief retrieves all elements from the hash table
   * \param[out] keys_out location to store retrieved keys
   * \param[out] values_out location to store corresponding retrieved values
   * \param[out] num_out number of of key/value pairs retrieved
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void retrieve_all(
    key_type* const keys_out,
    value_type* const values_out,
    index_type& num_out,
    cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    if (!is_initialized_) return;

    index_type* tmp = temp_.get();

    cudaMemsetAsync(tmp, 0, sizeof(index_t), stream);

    for_each(
      [=, *this] DEVICEQUALIFIER(key_type key, const value_type& value) {
        const auto i  = helpers::atomicAggInc(tmp);
        keys_out[i]   = key;
        values_out[i] = value;
      },
      stream);

    cudaMemcpyAsync(&num_out, tmp, sizeof(index_type), D2H);

    if (stream == cudaStreamDefault) { cudaStreamSynchronize(stream); }
  }

  /*! \brief erases a key from the hash table
   * \param[in] key_in key to erase from the hash table
   * \param[in] group cooperative group
   * \param[in] probing_length maximum number of probing attempts
   * \return status (per thread)
   */
  DEVICEQUALIFIER INLINEQUALIFIER status_type
  erase(const key_type key_in,
        const cg::thread_block_tile<cg_size()>& group,
        const index_type probing_length = defaults::probing_length()) noexcept
  {
    if (!is_initialized_) return status_type::not_initialized();

    if (!is_valid_key(key_in)) {
      status_->atomic_join(status_type::invalid_key());
      return status_type::invalid_key();
    }

    ProbingScheme iter(capacity(), probing_length, group);

    for (index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next()) {
      Key table_key       = table_[i].key;
      const bool hit      = (table_key == key_in);
      const auto hit_mask = group.ballot(hit);

      if (hit_mask) {
        const auto leader = ffs(hit_mask) - 1;

        if (group.thread_rank() == leader) { table_[i].key = tombstone_key(); }

        return status_type::none();
      }

      if (group.any(is_empty_key(table_key))) {
        // return status_type::none();
        return status_type::key_not_found();
      }
    }

    // return status_type::none();
    return status_type::probing_length_exceeded();
  }

  /*! \brief erases a set of keys from the hash table
   * \tparam StatusHandler handles returned per key (see \c status_handlers )
   * \param[in] keys_in pointer to keys to erase from the hash table
   * \param[in] num_in number of keys to erase
   * \param[in] probing_length maximum number of probing attempts
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[out] status_out status information (per key)
   */
  template <class StatusHandler = defaults::status_handler_t>
  HOSTQUALIFIER INLINEQUALIFIER void erase(
    const key_type* const keys_in,
    const index_type num_in,
    cudaStream_t stream                                 = cudaStreamDefault,
    const index_type probing_length                     = defaults::probing_length(),
    typename StatusHandler::base_type* const status_out = nullptr) noexcept
  {
    static_assert(checks::is_status_handler<StatusHandler>(), "not a valid status handler type");

    if (!is_initialized_) return;

    kernels::erase<SingleValueHashTable, StatusHandler>
      <<<SDIV(num_in * cg_size(), WARPCORE_BLOCKSIZE), WARPCORE_BLOCKSIZE, 0, stream>>>(
        keys_in, num_in, *this, probing_length, status_out);
  }

  /*! \brief applies a function over all key value pairs inside the table
   * \tparam Func type of map i.e. CUDA device lambda
   * \param[in] f map to apply
   * \param[in] stream CUDA stream in which this operation is executed in
   * \param[in] size of shared memory to reserve for this execution
   */
  template <class Func>
  HOSTQUALIFIER INLINEQUALIFIER void for_each(Func f,  // TODO const?
                                              cudaStream_t stream         = cudaStreamDefault,
                                              const index_type smem_bytes = 0) const noexcept
  {
    if (!is_initialized_) return;

    helpers::lambda_kernel<<<SDIV(capacity(), WARPCORE_BLOCKSIZE),
                             WARPCORE_BLOCKSIZE,
                             smem_bytes,
                             stream>>>([=, *this] DEVICEQUALIFIER {
      const index_type tid = helpers::global_thread_id();

      if (tid < capacity()) {
        auto&& pair = table_[tid];
        if (is_valid_key(pair.key)) { f(pair.key, pair.value); }
      }
    });
  }

  /*! \brief number of key/value pairs stored inside the hash table
   * \param[in] stream CUDA stream in which this operation is executed in
   * \return the number of key/value pairs inside the hash table
   */
  HOSTQUALIFIER INLINEQUALIFIER index_type
  size(cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    if (!is_initialized_) return 0;

    index_type out;
    index_type* tmp = temp_.get();

    cudaMemsetAsync(tmp, 0, sizeof(index_t), stream);

    helpers::lambda_kernel<<<SDIV(capacity(), WARPCORE_BLOCKSIZE),
                             WARPCORE_BLOCKSIZE,
                             sizeof(index_type),
                             stream>>>([=, *this] DEVICEQUALIFIER {
      __shared__ index_type smem;

      const index_type tid = helpers::global_thread_id();
      const auto block     = cg::this_thread_block();

      if (tid >= capacity()) return;

      const bool empty = !is_valid_key(table_[tid].key);

      if (block.thread_rank() == 0) { smem = 0; }

      block.sync();

      if (!empty) {
        const auto active_threads = cg::coalesced_threads();

        if (active_threads.thread_rank() == 0) { atomicAdd(&smem, active_threads.size()); }
      }

      block.sync();

      if (block.thread_rank() == 0 && smem != 0) { atomicAdd(tmp, smem); }
    });

    cudaMemcpyAsync(&out, tmp, sizeof(index_type), D2H, stream);

    cudaStreamSynchronize(stream);

    return out;
  }

  /*! \brief current load factor of the hash table
   * \param stream CUDA stream in which this operation is executed in
   * \return load factor
   */
  HOSTQUALIFIER INLINEQUALIFIER float load_factor(
    cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    return float(size(stream)) / float(capacity());
  }

  /*! \brief current storage density of the hash table
   * \param stream CUDA stream in which this operation is executed in
   * \return storage density
   */
  HOSTQUALIFIER INLINEQUALIFIER float storage_density(
    cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    return load_factor(stream);
  }

  /*! \brief get the capacity of the hash table
   * \return number of slots in the hash table
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER index_type capacity() const noexcept
  {
    return table_.capacity();
  }

  /*! \brief get the total number of bytes occupied by this data structure
   *  \return bytes
   */
  HOSTQUALIFIER INLINEQUALIFIER index_type bytes_total() const noexcept
  {
    return table_.bytes_total() + temp_.bytes_total() + sizeof(status_type);
  }

  /*! \brief indicates if the hash table is properly initialized
   * \return \c true iff the hash table is properly initialized
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER bool is_initialized() const noexcept
  {
    return is_initialized_;
  }

  /*! \brief get the status of the hash table
   * \param stream CUDA stream in which this operation is executed in
   * \return the status
   */
  HOSTQUALIFIER INLINEQUALIFIER status_type
  peek_status(cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    status_type status = status_type::not_initialized();

    if (status_ != nullptr) {
      cudaMemcpyAsync(&status, status_, sizeof(status_type), D2H, stream);

      cudaStreamSynchronize(stream);
    }

    return status;
  }

  /*! \brief get and reset the status of the hash table
   * \param[in] stream CUDA stream in which this operation is executed in
   * \return the status
   */
  HOSTQUALIFIER INLINEQUALIFIER status_type
  pop_status(cudaStream_t stream = cudaStreamDefault) noexcept
  {
    status_type status = status_type::not_initialized();

    if (status_ != nullptr) {
      cudaMemcpyAsync(&status, status_, sizeof(status_type), D2H, stream);

      assign_status(table_.status(), stream);
    }

    return status;
  }

  /*! \brief checks if \c key is equal to \c EmptyKey
   * \return \c bool
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr bool is_empty_key(
    const key_type key) noexcept
  {
    return (key == empty_key());
  }

  /*! \brief checks if \c key is equal to \c TombstoneKey
   * \return \c bool
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr bool is_tombstone_key(
    const key_type key) noexcept
  {
    return (key == tombstone_key());
  }

  /*! \brief checks if \c key is equal to \c (EmptyKey||TombstoneKey)
   * \return \c bool
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER static constexpr bool is_valid_key(
    const key_type key) noexcept
  {
    return (key != empty_key() && key != tombstone_key());
  }

  /*! \brief get random seed
   * \return seed
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER key_type seed() const noexcept { return seed_; }

  /*! \brief indicates if this object is a shallow copy
   * \return \c bool
   */
  HOSTDEVICEQUALIFIER INLINEQUALIFIER bool is_copy() const noexcept { return is_copy_; }

 private:
  /*! \brief internal insert implementation
   * \param[in] key_in key to insert into the hash table
   * \param[out] status_out status returned by this operation
   * \param[in] group cooperative group
   * \param[in] probing_length maximum number of probing attempts
   * \return pointer to the corresponding value slot
   */
  DEVICEQUALIFIER INLINEQUALIFIER value_type* insert_impl(
    const key_type key_in,
    status_type& status_out,
    const cg::thread_block_tile<cg_size()>& group,
    const index_type probing_length) noexcept
  {
    if (!is_initialized_) {
      status_out = status_type::not_initialized();
      return nullptr;
    }

    if (!is_valid_key(key_in)) {
      status_->atomic_join(status_type::invalid_key());
      status_out = status_type::invalid_key();
      return nullptr;
    }

    ProbingScheme iter(capacity(), probing_length, group);

    for (index_type i = iter.begin(key_in, seed_); i != iter.end(); i = iter.next()) {
      key_type table_key  = table_[i].key;
      const bool hit      = (table_key == key_in);
      const auto hit_mask = group.ballot(hit);

      if (hit_mask) {
        status_->atomic_join(status_type::duplicate_key());
        status_out = status_type::duplicate_key();

        const auto leader       = ffs(hit_mask) - 1;
        const auto leader_index = group.shfl(i, leader);
        return &(table_[leader_index].value);
      }

      // !not_is_valid_key?
      auto empty_mask = group.ballot(is_empty_key(table_key));

      bool success   = false;
      bool duplicate = false;

      while (empty_mask) {
        const auto leader = ffs(empty_mask) - 1;

        if (group.thread_rank() == leader) {
          const auto old = atomicCAS(&(table_[i].key), table_key, key_in);

          success   = (old == table_key);
          duplicate = (old == key_in);
        }

        if (group.any(duplicate)) {
          status_->atomic_join(status_type::duplicate_key());
          status_out = status_type::duplicate_key();

          const auto leader_index = group.shfl(i, leader);
          return &(table_[leader_index].value);
        }

        if (group.any(success)) {
          status_out              = status_type::none();
          const auto leader_index = group.shfl(i, leader);

          return &(table_[leader_index].value);
        }

        empty_mask ^= 1UL << leader;
      }
    }

    status_->atomic_join(status_type::probing_length_exceeded());
    status_out = status_type::probing_length_exceeded();
    return nullptr;
  }

  /*! \brief assigns the hash table's status
   * \param[in] status new status
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void assign_status(
    const status_type status, cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    if (status_ != nullptr) {
      cudaMemcpyAsync(status_, &status, sizeof(status_type), H2D, stream);

      cudaStreamSynchronize(stream);
    }
  }

  /*! \brief joins additional flags to the hash table's status
   * \param[in] status new status
   * \param[in] stream CUDA stream in which this operation is executed in
   */
  HOSTQUALIFIER INLINEQUALIFIER void join_status(
    const status_type status, cudaStream_t stream = cudaStreamDefault) const noexcept
  {
    if (status_ != nullptr) {
      const status_type joined = peek_status(stream) + status;

      cudaMemcpyAsync(status_, &joined, sizeof(status_type), H2D, stream);

      cudaStreamSynchronize(stream);
    }
  }

  /*! \brief joins additional flags to the hash table's status
   * \info \c const on purpose
   * \param[in] status new status
   */
  DEVICEQUALIFIER INLINEQUALIFIER void device_join_status(const status_type status) const noexcept
  {
    if (status_ != nullptr) { status_->atomic_join(status); }
  }

  status_type* status_;  //< pointer to status
  TableStorage table_;   //< actual key/value storage
  temp_type temp_;       //< temporary memory
  key_type seed_;        //< random seed
  bool is_initialized_;  //< indicates if table is properly initialized
  bool is_copy_;         //< indicates if table is a shallow copy

  // friend declarations
  template <class Key_,
            class Value_,
            Key_ EmptyKey_,
            Key_ TombstoneKey_,
            class ProbingScheme_,
            class TableStorage_,
            index_type TempMemoryBytes_>
  friend class CountingHashTable;

  template <class Key_,
            class Value_,
            Key_ EmptyKey_,
            Key_ TombstoneKey_,
            class ValueStore_,
            class ProbingScheme_>
  friend class BucketListHashTable;

};  // class SingleValueHashTable

}  // namespace warpcore

#endif /* WARPCORE_SINGLE_VALUE_HASH_TABLE_CUH */