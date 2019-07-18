#include "activations.h"

#include "constants.h"
#include "types.h"

template <const int lstm_state_size>
void relu(FDATA_T* input_feature_map, FDATA_T* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    output_feature_map[result_idx] = input_feature_map[result_idx] > FDATA_T(0)?
                                     input_feature_map[result_idx] : FDATA_T(0);
  }
}

// template <>
// void softmax (FDATA_T* input_feature_map,
              // FDATA_T* output_probability_distribution) {
  // // for fixed length array (FC_OUTPUT_SIZE)

  // // compute denominator
  // FDATA_T denominator = 0;
  // for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++) {
    // denominator += exp(input_feature_map[i]);
  // }

  // // compute probability distribution
  // for (LDATA_T result_idx = 0; result_idx < FC_OUTPUT_SIZE; result_idx++) {
    // output_probability_distribution[result_idx] =
        // exp(input_feature_map[result_idx]) / denominator;
  // }
// }
