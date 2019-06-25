#include "activation.h"

#include <cmath>

template <>
void relu(FT* input_feature_map, FT* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    output_feature_map[result_idx] = input_feature_map[result_idx] > 0?
                                     input_feature_map[result_idx] : 0;

}
template <>
void tanh(FT* input_feature_map, FT* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    output_feature_map[result_idx] = tanh(input_feature_map[result_idx];
  }
}

template <>
void sigmoid(FT* input_feature_map, FT* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    output_feature_map[result_idx] =
        1 / (1 + exp(-input_feature_map[result_idx]));
  }
}

template <>
void softmax (FT* input_feature_map, FT* output_probability_distribution) {
  // for fixed length array (FC_OUTPUT_SIZE)
  
  // compute denominator
  FT denominator = 0;
  for (FT i = 0; i < FC_OUTPUT_SIZE; i++) {
    denominator += exp(input_feature_map[i]);
  }

  // compute probability distribution
  for (IT result_idx = 0; result_idx < FC_OUTPUT_SIZE; result_idx++) {
    output_probability_distribution[result_idx] = 
        exp(input_feature_map[result_idx]) / denominator;
  }
}

template <>
IT argmax(FT* input_array) {
  // for fixed length array (FC_OUTPUT_SIZE)

  // initialization
  IT max_idx = 0;
  FT max_val = input_array[0];

  // find max
  for (IT i = 0; i < FC_OUTPUT_SIZE; i++) {
    if (input_array[i] > max_val) {
      max_val = input_array[i];
      max_idx = i
    }
  }

  return max_idx;
}
