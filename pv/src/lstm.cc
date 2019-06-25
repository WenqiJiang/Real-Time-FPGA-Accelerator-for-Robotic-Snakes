#include "lstm.h"

#include <cmath>

#include "activations.h"
#include "utils.h"

template <>
void lstm(FT* forget_gate_kernel_last_state,
          FT* forget_gate_kernel_input_state, FT* forget_gate_bias,
          FT* input_gate_kernel_last_state,
          FT* input_gate_kernel_input_state, FT* input_gate_bias,
          FT* tanh_gate_kernel_last_state, 
          FT* tanh_gate_kernel_input_state, FT* tanh_gate_bias,
          FT* output_gate_kernel_last_state,
          FT* output_gate_kernel_input_state, FT* output_gate_bias,
          FT* lstm_last_state, FT* lstm_input_state, FT* last_candidate,
          FT* forget_gate_last_candidate_mul_cache, 
          FT* input_gate_tanh_gate_mul_cache,
          FT* tanh_new_candidate_cache,
          FT* new_candidate, FT* lstm_output_state) {
  // trick: intermediate results are inputs as well, they are malloced outside
  //  LSTM block, so that we don't need to malloc they in every single timestep.
  //  This will improve the software performance
  // input: kernels and biases of 4 gates
  //  xx_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE
  //  xx_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE
  //  xx_gate_bias: LSTM_STATE_SIZE
  // intermediate result (cache): forget_gate_last_candidate_mul_cache,
  //  input_gate_tanh_gate_mul_cache, tanh_new_candidate_cache, 
  // output: new_candidate and lstm_output_state,
  //  both with size of LSTM_STATE_SIZE

  // 4 gates
  forget_gate<FT>(
      forget_gate_kernel_last_state, forget_gate_kernel_input_state, 
      forget_gate_bias, lstm_last_state, lstm_input_state, forget_gate_result);

  input_gate<FT>(
      input_gate_kernel_last_state, input_gate_kernel_input_state, 
      input_gate_bias, lstm_last_state, lstm_input_state, input_gate_result);

  tanh_gate<FT>(
      tanh_gate_kernel_last_state, tanh_gate_kernel_input_state,
      tanh_gate_bias, lstm_last_state, lstm_input_state, tanh_gate_result);

  output_gate<FT>(
      output_gate_kernel_last_state, output_gate_kernel_input_state, 
      output_gate_bias, lstm_last_state, lstm_input_state, output_gate_result);

  // elementwise mul
  elementwise_mul<FT>(forget_gate_result, last_candidate,
                           forget_gate_last_candidate_mul_cache);
  elementwise_mul<FT>(input_gate_result, tanh_gate_result,
                           input_gate_tanh_gate_mul_cache);

  // compute new candidate
  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    new_candidate[result_idx] = 
        forget_gate_last_candidate_mul_cache[result_idx] + 
        input_gate_tanh_gate_mul_cache;
  }

  // tanh new candidate
  tanh<FT>(new_candidate, tanh_new_candidate_cache);

  // compute output state
  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    lstm_output_state[result_idx] = tanh_new_candidate_cache[result_idx] *
                               output_gate_result[result_idx];
  }
}

template <>
void forget_gate(FT* forget_gate_kernel_last_state,
                 FT* forget_gate_kernel_input_state, FT* forget_gate_bias,
                 FT* lstm_last_state, FT* lstm_input_state, 
                 FT* forget_gate_result) {
  // input:
  // forget_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE, 
  //  notice that this kernel is transposed, i.e. lstm_last_state do 
  //  multiplication with each single row instead of each single column
  // forget_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE,
  // notice that this kernel is transposed
  // forget_gate_bias: LSTM_STATE_SIZE
  // lstm_last_state: a single state, no batch, LSTM_STATE_SIZE
  // lstm_input_state: a single state, no batch, LSTM_INPUT_SIZE
  // output:
  // forget_gate_result: LSTM_STATE_SIZE

  gate_template<FT>(forget_gate_kernel_laste_state, 
                    forget_gate_kernel_input_state, forget_gate_bias,
                    lstm_last_state, lstm_input_state, forget_gate_result);

  sigmoid<FT>(forget_gate_result, forget_gate_result);
}

template <>
void input_gate(FT* input_gate_kernel_last_state,
                FT* input_gate_kernel_input_state, FT* input_gate_bias,
                FT* lstm_last_state, FT* lstm_input_state, 
                FT* input_gate_result) {
  // input:
  // input_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE, 
  //  notice that this kernel is transposed, i.e. lstm_last_state do 
  //  multiplication with each single row instead of each single column
  // input_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE,
  // notice that this kernel is transposed
  // input_gate_bias: LSTM_STATE_SIZE
  // lstm_last_state: a single state, no batch, LSTM_STATE_SIZE
  // lstm_input_state: a single state, no batch, LSTM_INPUT_SIZE
  // output:
  // input_gate_result: LSTM_STATE_SIZE

  gate_template<FT>(input_gate_kernel_laste_state, 
                    input_gate_kernel_input_state, input_gate_bias,
                    lstm_last_state, lstm_input_state, input_gate_result);

  sigmoid<FT>(input_gate_result, input_gate_result);
}

template <>
void tanh_gate(FT* tanh_gate_kernel_last_state,
               FT* tanh_gate_kernel_input_state, FT* tanh_gate_bias,
               FT* lstm_last_state, FT* lstm_input_state, 
               FT* tanh_gate_result) {
  // input:
  // tanh_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE, 
  //  notice that this kernel is transposed, i.e. lstm_last_state do 
  //  multiplication with each single row instead of each single column
  // tanh_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE,
  // notice that this kernel is transposed
  // tanh_gate_bias: LSTM_STATE_SIZE
  // lstm_last_state: a single state, no batch, LSTM_STATE_SIZE
  // lstm_input_state: a single state, no batch, LSTM_INPUT_SIZE
  // output:
  // tanh_gate_result: LSTM_STATE_SIZE

  gate_template<FT>(tanh_gate_kernel_laste_state, 
                    tanh_gate_kernel_input_state, tanh_gate_bias,
                    lstm_last_state, lstm_input_state, tanh_gate_result);

  tanh<FT>(tanh_gate_result, tanh_gate_result);
}

template <>
void output_gate(FT* output_gate_kernel_last_state,
                 FT* output_gate_kernel_input_state, FT* output_gate_bias,
                 FT* lstm_last_state, FT* lstm_input_state, 
                 FT* output_gate_result) {
  // input:
  // output_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE, 
  //  notice that this kernel is transposed, i.e. lstm_last_state do 
  //  multiplication with each single row instead of each single column
  // output_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE,
  // notice that this kernel is transposed
  // output_gate_bias: LSTM_STATE_SIZE
  // lstm_last_state: a single state, no batch, LSTM_STATE_SIZE
  // lstm_input_state: a single state, no batch, LSTM_INPUT_SIZE
  // output:
  // output_gate_result: LSTM_STATE_SIZE

  gate_template<FT>(output_gate_kernel_laste_state, 
                    output_gate_kernel_input_state, output_gate_bias,
                    lstm_last_state, lstm_input_state, output_gate_result);

  sigmoid<FT>(output_gate_result, output_gate_result);
}

template <>
void gate_template(FT* kernel_last_state, FT* kernel_input_state, FT* bias,
                   FT* lstm_last_state, FT* lstm_input_state, FT* result) {

  // input:
  // kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE, 
  //  notice that this kernel is transposed, i.e. lstm_last_state do 
  //  multiplication with each single row instead of each single column
  // kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE,
  // notice that this kernel is transposed
  // bias: LSTM_STATE_SIZE
  // lstm_last_state: a single state, no batch, LSTM_STATE_SIZE
  // lstm_input_state: a single state, no batch, LSTM_INPUT_SIZE
  // output:
  // result: LSTM_STATE_SIZE

  // initialization
  zero_init<FT>(result, LSTM_STATE_SIZE);

  // last state
  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    
    for (IT sum_idx = 0; sum_idx < LSTM_STATE_SIZE; sum_idx++) {

      IT kernel_last_state_idx = result_idx * LSTM_STATE_SIZE + sum_idx;

      result[result_idx] += 
          kernel_last_state[kernel_last_state_idx] * lstm_last_state[sum_idx];
    }
  }

  // input state
  for (IT result_idx = 0; result_idx < LSTM_INPUT_SIZE; result_idx++) {
    
    for (IT sum_idx = 0; sum_idx < LSTM_INPUT_SIZE; sum_idx++) {

      IT kernel_input_state_idx = result_idx * LSTM_INPUT_SIZE + sum_idx;

      result[result_idx] += 
          kernel_input_state[kernel_input_state_idx]*lstm_input_state[sum_idx];
    }
  }

  // bias
  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    result[result_idx] += bias[result_idx];
  }
}
template <>
void elementwise_mul(FT* input_vector1, FT* input_vector2, FT* output_vector) {
  // input: input_vectori, with a size of LSTM_STATE_SIZE
  // output: output_vector, with a size of LSTM_STATE_SIZE

  for (IT result_idx = 0; result_idx < LSTM_STATE_SIZE; result_idx++) {
    output_vector[result_idx] =
        input_vector1[result_idx] * input_vector2[result_idx];
  }
}
