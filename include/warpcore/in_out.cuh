namespace warpcore {

/*! \brief simple input-output wrapper for columns to help with variadic templates for variable
 * number of columns to read/write
 */
template <typename ValueType>
struct InOutColumnWrapper {
  ValueType* in;
  ValueType* out;
  InOutColumnWrapper(ValueType* input_values, ValueType* output_values)
    : in{input_values}, out{output_values} {};
};
}  // namespace warpcore