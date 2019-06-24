#pragma once

// for fixed length array (LSTM_SIZE)
template <typename FT>
void relu(FT* input_feature_map);

// for fixed length array (LSTM_SIZE)
template <typename FT>
void tanh(FT* input_feature_map);

// for fixed length array (LSTM_SIZE)
template <typename FT>
void sigmoid(FT* input_feature_map);

// for fixed length array (LSTM_SIZE)
template <typename FT>
void softmax (FT* input_feature_map, FT* output_probability_distribution);

// for fixed length array (LSTM_SIZE)
template <typename FT, typename IT>
void argmax(FT* input, IT result);
