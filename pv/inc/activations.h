#pragma once

#include "types.h"

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <typename FT>
void relu(FT* input_feature_map, FT* output_feature_map);

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <typename FT>
void tanh(FT* input_feature_map, FT* output_feature_map);

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <typename FT>
void sigmoid(FT* input_feature_map, FT* output_feature_map);

// for fixed length array (FC_OUTPUT_SIZE)
template <typename FT>
void softmax (FT* input_feature_map, FT* output_probability_distribution);

// for fixed length array (FC_OUTPUT_SIZE)
template <typename FT, typename LT>
LT argmax(FT* input_array);
