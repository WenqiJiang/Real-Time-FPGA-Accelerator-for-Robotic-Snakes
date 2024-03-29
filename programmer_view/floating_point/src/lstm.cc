#include "lstm.h"

#include <cmath>

#include "activations.h"
#include "constants.h"
#include "types.h"
#include "utils.h"

template <const int lstm_state_size, const int lstm_input_size>
void gate_template(const FDATA_T kernel_last_state
                       [lstm_state_size * lstm_state_size],
                   const FDATA_T kernel_input_state
                       [lstm_state_size * lstm_input_size],
                   const FDATA_T bias[lstm_state_size],
                   const FDATA_T lstm_last_state[lstm_state_size],
                   const FDATA_T lstm_input_state[lstm_input_size],
                   FDATA_T result[lstm_state_size]) {

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
  zero_init<FDATA_T, LDATA_T>(result, lstm_state_size);

  // last state
  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {

    for (LDATA_T sum_idx = 0; sum_idx < lstm_state_size; sum_idx++) {

      LDATA_T kernel_last_state_idx = result_idx * lstm_state_size + sum_idx;

      result[result_idx] +=
          kernel_last_state[kernel_last_state_idx] * lstm_last_state[sum_idx];
    }
  }

  // input state
  for (LDATA_T result_idx = 0; result_idx < lstm_input_size; result_idx++) {

    for (LDATA_T sum_idx = 0; sum_idx < lstm_input_size; sum_idx++) {

      LDATA_T kernel_input_state_idx = result_idx * lstm_input_size + sum_idx;

      result[result_idx] +=
          kernel_input_state[kernel_input_state_idx]*lstm_input_state[sum_idx];
    }
  }

  // bias
  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    result[result_idx] += bias[result_idx];
  }
}

template <const int lstm_state_size, const int lstm_input_size>
void forget_gate(const FDATA_T forget_gate_kernel_last_state
                     [lstm_state_size * lstm_state_size],
                 const FDATA_T forget_gate_kernel_input_state
                     [lstm_state_size * lstm_input_size],
                 const FDATA_T forget_gate_bias[lstm_state_size],
                 const FDATA_T lstm_last_state[lstm_state_size],
                 const FDATA_T lstm_input_state[lstm_input_size],
                 FDATA_T forget_gate_result[lstm_state_size]) {
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

  gate_template<lstm_state_size, lstm_input_size>(
      forget_gate_kernel_last_state, forget_gate_kernel_input_state,
      forget_gate_bias, lstm_last_state, lstm_input_state, forget_gate_result);

  sigmoid<lstm_state_size>(forget_gate_result, forget_gate_result);
}

template <const int lstm_state_size, const int lstm_input_size>
void input_gate(const FDATA_T input_gate_kernel_last_state
                    [lstm_state_size * lstm_state_size],
                const FDATA_T input_gate_kernel_input_state
                    [lstm_state_size * lstm_input_size],
                const FDATA_T input_gate_bias[lstm_state_size],
                const FDATA_T lstm_last_state[lstm_state_size],
                const FDATA_T lstm_input_state[lstm_input_size],
                FDATA_T input_gate_result[lstm_state_size]) {
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

  gate_template<lstm_state_size, lstm_input_size>(
      input_gate_kernel_last_state, input_gate_kernel_input_state,
      input_gate_bias, lstm_last_state, lstm_input_state, input_gate_result);

  sigmoid<lstm_state_size>(input_gate_result, input_gate_result);
}

template <const int lstm_state_size, const int lstm_input_size>
void tanh_gate(const FDATA_T tanh_gate_kernel_last_state
                   [lstm_state_size * lstm_state_size],
               const FDATA_T tanh_gate_kernel_input_state
                   [lstm_state_size * lstm_input_size],
               const FDATA_T tanh_gate_bias[lstm_state_size],
               const FDATA_T lstm_last_state[lstm_state_size],
               const FDATA_T lstm_input_state[lstm_input_size],
               FDATA_T tanh_gate_result[lstm_state_size]) {
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

  gate_template<lstm_state_size, lstm_input_size>(
      tanh_gate_kernel_last_state, tanh_gate_kernel_input_state,
      tanh_gate_bias, lstm_last_state, lstm_input_state, tanh_gate_result);

  tanh<lstm_state_size>(tanh_gate_result, tanh_gate_result);
}

template <const int lstm_state_size, const int lstm_input_size>
void output_gate(const FDATA_T output_gate_kernel_last_state
                     [lstm_state_size * lstm_state_size],
                 const FDATA_T output_gate_kernel_input_state
                     [lstm_state_size * lstm_input_size],
                 const FDATA_T output_gate_bias[lstm_state_size],
                 const FDATA_T lstm_last_state[lstm_state_size],
                 const FDATA_T lstm_input_state[lstm_input_size],
                 FDATA_T output_gate_result[lstm_state_size]) {
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

  gate_template<lstm_state_size, lstm_input_size>(
      output_gate_kernel_last_state, output_gate_kernel_input_state,
      output_gate_bias, lstm_last_state, lstm_input_state, output_gate_result);

  sigmoid<lstm_state_size>(output_gate_result, output_gate_result);
}

template <const int lstm_state_size>
void elementwise_mul(const FDATA_T input_vector1[lstm_state_size],
                     const FDATA_T input_vector2[lstm_state_size],
                     FDATA_T output_vector[lstm_state_size]) {
  // input: input_vectori, with a size of LSTM_STATE_SIZE
  // output: output_vector, with a size of LSTM_STATE_SIZE

  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    output_vector[result_idx] =
        input_vector1[result_idx] * input_vector2[result_idx];
  }
}
template <const int lstm_state_size, const int lstm_input_size>
void lstm(const FDATA_T forget_gate_kernel_last_state
              [lstm_state_size * lstm_state_size],
          const FDATA_T forget_gate_kernel_input_state
              [lstm_state_size * lstm_input_size],
          const FDATA_T forget_gate_bias[lstm_state_size],
          const FDATA_T input_gate_kernel_last_state
              [lstm_state_size * lstm_state_size],
          const FDATA_T input_gate_kernel_input_state
              [lstm_state_size * lstm_input_size],
          const FDATA_T input_gate_bias[lstm_state_size],
          const FDATA_T tanh_gate_kernel_last_state
              [lstm_state_size * lstm_state_size],
          const FDATA_T tanh_gate_kernel_input_state
              [lstm_state_size * lstm_input_size],
          const FDATA_T tanh_gate_bias[lstm_state_size],
          const FDATA_T output_gate_kernel_last_state
              [lstm_state_size * lstm_state_size],
          const FDATA_T output_gate_kernel_input_state
              [lstm_state_size * lstm_input_size],
          const FDATA_T output_gate_bias[lstm_state_size],
          const FDATA_T lstm_last_state[lstm_state_size],
          const FDATA_T lstm_input_state[lstm_input_size],
          const FDATA_T last_candidate[lstm_state_size],
          FDATA_T forget_gate_result[lstm_state_size],
          FDATA_T input_gate_result[lstm_state_size],
          FDATA_T tanh_gate_result[lstm_state_size],
          FDATA_T output_gate_result[lstm_state_size],
          FDATA_T forget_gate_last_candidate_mul_cache[lstm_state_size],
          FDATA_T input_gate_tanh_gate_mul_cache[lstm_state_size],
          FDATA_T tanh_new_candidate_cache[lstm_state_size],
          FDATA_T new_candidate[lstm_state_size],
          FDATA_T lstm_output_state[lstm_state_size]) {
  // trick: intermediate results are inputs as well, they are malloced outside
  //  LSTM block, so that we don't need to malloc they in every single timestep.
  //  This will improve the software performance
  // input: kernels and biases of 4 gates
  //  xx_gate_kernel_last_state: LSTM_STATE_SIZE * LSTM_STATE_SIZE
  //  xx_gate_kernel_input_state: LSTM_STATE_SIZE * LSTM_INPUT_SIZE
  //  xx_gate_bias: LSTM_STATE_SIZE
  // intermediate result (cache):
  //  forget, intput, tanh, output gate results
  //  forget_gate_last_candidate_mul_cache,
  //  input_gate_tanh_gate_mul_cache, tanh_new_candidate_cache,
  // output: new_candidate and lstm_output_state,
  //  both with size of LSTM_STATE_SIZE

  // 4 gates
  forget_gate<lstm_state_size, lstm_input_size>(
      forget_gate_kernel_last_state, forget_gate_kernel_input_state,
      forget_gate_bias, lstm_last_state, lstm_input_state, forget_gate_result);

  input_gate<lstm_state_size, lstm_input_size>(
      input_gate_kernel_last_state, input_gate_kernel_input_state,
      input_gate_bias, lstm_last_state, lstm_input_state, input_gate_result);

  tanh_gate<lstm_state_size, lstm_input_size>(
      tanh_gate_kernel_last_state, tanh_gate_kernel_input_state,
      tanh_gate_bias, lstm_last_state, lstm_input_state, tanh_gate_result);

  output_gate<lstm_state_size, lstm_input_size>(
      output_gate_kernel_last_state, output_gate_kernel_input_state,
      output_gate_bias, lstm_last_state, lstm_input_state, output_gate_result);

  // elementwise mul
  elementwise_mul<lstm_state_size>(
      forget_gate_result, last_candidate, forget_gate_last_candidate_mul_cache);

  elementwise_mul<lstm_state_size>(
      input_gate_result, tanh_gate_result, input_gate_tanh_gate_mul_cache);

  // compute new candidate
  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    new_candidate[result_idx] =
        forget_gate_last_candidate_mul_cache[result_idx] +
        input_gate_tanh_gate_mul_cache[result_idx];
  }

  // tanh new candidate
  tanh<lstm_state_size>(new_candidate, tanh_new_candidate_cache);

  // compute output state
  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    lstm_output_state[result_idx] = tanh_new_candidate_cache[result_idx] *
                               output_gate_result[result_idx];
  }
}

// instantiation
template void lstm<LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>(
    const FDATA_T forget_gate_kernel_last_state
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T forget_gate_kernel_input_state
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T forget_gate_bias[LSTM_STATE_SIZE_1],
    const FDATA_T input_gate_kernel_last_state
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T input_gate_kernel_input_state
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T input_gate_bias[LSTM_STATE_SIZE_1],
    const FDATA_T tanh_gate_kernel_last_state
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T tanh_gate_kernel_input_state
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T tanh_gate_bias[LSTM_STATE_SIZE_1],
    const FDATA_T output_gate_kernel_last_state
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T output_gate_kernel_input_state
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T output_gate_bias[LSTM_STATE_SIZE_1],
    const FDATA_T lstm_last_state[LSTM_STATE_SIZE_1],
    const FDATA_T lstm_input_state[LSTM_INPUT_SIZE_1],
    const FDATA_T last_candidate[LSTM_STATE_SIZE_1],
    FDATA_T forget_gate_result[LSTM_STATE_SIZE_1],
    FDATA_T input_gate_result[LSTM_STATE_SIZE_1],
    FDATA_T tanh_gate_result[LSTM_STATE_SIZE_1],
    FDATA_T output_gate_result[LSTM_STATE_SIZE_1],
    FDATA_T forget_gate_last_candidate_mul_cache[LSTM_STATE_SIZE_1],
    FDATA_T input_gate_tanh_gate_mul_cache[LSTM_STATE_SIZE_1],
    FDATA_T tanh_new_candidate_cache[LSTM_STATE_SIZE_1],
    FDATA_T new_candidate[LSTM_STATE_SIZE_1],
    FDATA_T lstm_output_state[LSTM_STATE_SIZE_1]);

template void lstm<LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>(
    const FDATA_T forget_gate_kernel_last_state
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T forget_gate_kernel_input_state
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T forget_gate_bias[LSTM_STATE_SIZE_2],
    const FDATA_T input_gate_kernel_last_state
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T input_gate_kernel_input_state
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T input_gate_bias[LSTM_STATE_SIZE_2],
    const FDATA_T tanh_gate_kernel_last_state
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T tanh_gate_kernel_input_state
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T tanh_gate_bias[LSTM_STATE_SIZE_2],
    const FDATA_T output_gate_kernel_last_state
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T output_gate_kernel_input_state
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T output_gate_bias[LSTM_STATE_SIZE_2],
    const FDATA_T lstm_last_state[LSTM_STATE_SIZE_2],
    const FDATA_T lstm_input_state[LSTM_INPUT_SIZE_2],
    const FDATA_T last_candidate[LSTM_STATE_SIZE_2],
    FDATA_T forget_gate_result[LSTM_STATE_SIZE_2],
    FDATA_T input_gate_result[LSTM_STATE_SIZE_2],
    FDATA_T tanh_gate_result[LSTM_STATE_SIZE_2],
    FDATA_T output_gate_result[LSTM_STATE_SIZE_2],
    FDATA_T forget_gate_last_candidate_mul_cache[LSTM_STATE_SIZE_2],
    FDATA_T input_gate_tanh_gate_mul_cache[LSTM_STATE_SIZE_2],
    FDATA_T tanh_new_candidate_cache[LSTM_STATE_SIZE_2],
    FDATA_T new_candidate[LSTM_STATE_SIZE_2],
    FDATA_T lstm_output_state[LSTM_STATE_SIZE_2]);
