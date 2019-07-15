#pragma once

#include "constants.h"
#include "types.h"

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <const int lstm_state_size>
void relu(FDATA_T* input_feature_map, FDATA_T* output_feature_map);

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <const int lstm_state_size>
void tanh(FDATA_T* input_feature_map, FDATA_T* output_feature_map);

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <const int lstm_state_size>
void sigmoid(FDATA_T* input_feature_map, FDATA_T* output_feature_map);

// for fixed length array (FC_OUTPUT_SIZE)
template <typename FT>
void softmax (FT* input_feature_map, FT* output_probability_distribution);

// for fixed length array (FC_OUTPUT_SIZE)
template <typename FT, typename LT>
LT argmax(FT* input_array);

