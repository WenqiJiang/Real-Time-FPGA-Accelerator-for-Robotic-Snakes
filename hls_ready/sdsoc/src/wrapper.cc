#include "wrapper.h"

#include <cstdlib>

#include "constants.h"
#include "types.h"

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

#pragma SDS data zero_copy(forget_gate_kernel_last_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(forget_gate_kernel_input_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1])
#pragma SDS data zero_copy(forget_gate_bias_1 \
      [0: LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(input_gate_kernel_last_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(input_gate_kernel_input_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1])
#pragma SDS data zero_copy(input_gate_bias_1 \
      [0: LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(tanh_gate_kernel_last_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(tanh_gate_kernel_input_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1])
#pragma SDS data zero_copy(tanh_gate_bias_1 \
      [0: LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(output_gate_kernel_last_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1])
#pragma SDS data zero_copy(output_gate_kernel_input_state_1 \
      [0: LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1])
#pragma SDS data zero_copy(output_gate_bias_1 \
      [0: LSTM_STATE_SIZE_1])

  // LSTM Layer 2 weights
#pragma SDS data zero_copy(forget_gate_kernel_last_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(forget_gate_kernel_input_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2])
#pragma SDS data zero_copy(forget_gate_bias_2 \
      [0: LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(input_gate_kernel_last_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(input_gate_kernel_input_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2])
#pragma SDS data zero_copy(input_gate_bias_2 \
      [0: LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(tanh_gate_kernel_last_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(tanh_gate_kernel_input_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2])
#pragma SDS data zero_copy(tanh_gate_bias_2 \
      [0: LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(output_gate_kernel_last_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2])
#pragma SDS data zero_copy(output_gate_kernel_input_state_2 \
      [0: LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2])
#pragma SDS data zero_copy(output_gate_bias_2 \
      [0: LSTM_STATE_SIZE_2])

// Layer 1, a super large array, contain the inputs of all steps
#pragma SDS data zero_copy(lstm_input_state_1 \
      [0: LSTM_INPUT_SIZE_1 * COMPUTE_TIME])

// fc weights
#pragma SDS data zero_copy(fc_kernel[0: FC_OUTPUT_SIZE * FC_INPUT_SIZE])
#pragma SDS data zero_copy(fc_bias[0: FC_OUTPUT_SIZE])

#pragma SDS data zero_copy(results[0: COMPUTE_TIME])

// // data access pattern
// #pragma SDS data access_pattern( \
  // word_embedding: SEQUENTIAL, \
  // rnn_kernel: SEQUENTIAL, \
  // rnn_recurrent_kernel: SEQUENTIAL, \
  // rnn_bias: SEQUENTIAL, \
  // fc_kernel: SEQUENTIAL, \
  // fc_bias: SEQUENTIAL, \
  // rnn_init_state: SEQUENTIAL, \
  // rnn_init_idx: SEQUENTIAL, \
  // result_idx_all: SEQUENTIAL)

void wrapper_inference(
    const FDATA_T forget_gate_kernel_last_state_1
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T forget_gate_kernel_input_state_1
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T forget_gate_bias_1[LSTM_STATE_SIZE_1],
    const FDATA_T input_gate_kernel_last_state_1
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T input_gate_kernel_input_state_1
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T input_gate_bias_1[LSTM_STATE_SIZE_1],
    const FDATA_T tanh_gate_kernel_last_state_1
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T tanh_gate_kernel_input_state_1
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T tanh_gate_bias_1[LSTM_STATE_SIZE_1],
    const FDATA_T output_gate_kernel_last_state_1
        [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T output_gate_kernel_input_state_1
        [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T output_gate_bias_1[LSTM_STATE_SIZE_1],
    const FDATA_T forget_gate_kernel_last_state_2
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T forget_gate_kernel_input_state_2
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T forget_gate_bias_2[LSTM_STATE_SIZE_2],
    const FDATA_T input_gate_kernel_last_state_2
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T input_gate_kernel_input_state_2
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T input_gate_bias_2[LSTM_STATE_SIZE_2],
    const FDATA_T tanh_gate_kernel_last_state_2
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T tanh_gate_kernel_input_state_2
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T tanh_gate_bias_2[LSTM_STATE_SIZE_2],
    const FDATA_T output_gate_kernel_last_state_2
        [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T output_gate_kernel_input_state_2
        [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T output_gate_bias_2[LSTM_STATE_SIZE_2],
    const FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
    const FDATA_T fc_bias[FC_OUTPUT_SIZE],
    const FDATA_T lstm_input_state_1[LSTM_INPUT_SIZE_1 * COMPUTE_TIME],
    IDATA_T results[COMPUTE_TIME]) {

  // lstm caches (reuse this case in lstm cells, avoid malloc repeatly
  // Layer 1
  FDATA_T forget_gate_result_1[LSTM_STATE_SIZE_1];
  FDATA_T input_gate_result_1[LSTM_STATE_SIZE_1];
  FDATA_T tanh_gate_result_1[LSTM_STATE_SIZE_1];
  FDATA_T output_gate_result_1[LSTM_STATE_SIZE_1];
  FDATA_T forget_gate_last_candidate_mul_cache_1[LSTM_STATE_SIZE_1];
  FDATA_T input_gate_tanh_gate_mul_cache_1[LSTM_STATE_SIZE_1];
  FDATA_T tanh_new_candidate_cache_1[LSTM_STATE_SIZE_1];

  // Layer 2
  FDATA_T forget_gate_result_2[LSTM_STATE_SIZE_2];
  FDATA_T input_gate_result_2[LSTM_STATE_SIZE_2];
  FDATA_T tanh_gate_result_2[LSTM_STATE_SIZE_2];
  FDATA_T output_gate_result_2[LSTM_STATE_SIZE_2];
  FDATA_T forget_gate_last_candidate_mul_cache_2[LSTM_STATE_SIZE_2];
  FDATA_T input_gate_tanh_gate_mul_cache_2[LSTM_STATE_SIZE_2];
  FDATA_T tanh_new_candidate_cache_2[LSTM_STATE_SIZE_2];

  // LSTM states
  // ping-pong
  FDATA_T lstm_state1_1[LSTM_STATE_SIZE_1];
  FDATA_T lstm_state2_1[LSTM_STATE_SIZE_1];
  FDATA_T lstm_candidate1_1[LSTM_STATE_SIZE_1];
  FDATA_T lstm_candidate2_1[LSTM_STATE_SIZE_1];

  // Layer 2
  FDATA_T lstm_state1_2[LSTM_STATE_SIZE_2];
  FDATA_T lstm_state2_2[LSTM_STATE_SIZE_2];
  FDATA_T lstm_candidate1_2[LSTM_STATE_SIZE_2];
  FDATA_T lstm_candidate2_2[LSTM_STATE_SIZE_2];

  // fc output
  FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE];

  // ping-pong
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME / 2;
       compute_time++) {

    // ping
    lstm<LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>(
        forget_gate_kernel_last_state_1, forget_gate_kernel_input_state_1,
        forget_gate_bias_1, input_gate_kernel_last_state_1,
        input_gate_kernel_input_state_1, input_gate_bias_1,
        tanh_gate_kernel_last_state_1, tanh_gate_kernel_input_state_1,
        tanh_gate_bias_1, output_gate_kernel_last_state_1,
        output_gate_kernel_input_state_1, output_gate_bias_1,
        /* lstm_last_state = */ lstm_state1_1,
        /* lstm_input_state = */ lstm_input_state_1 +
        LSTM_INPUT_SIZE_1 * compute_time * 2,
        /* lstm_last_candidate = */ lstm_candidate1_1, forget_gate_result_1,
        input_gate_result_1, tanh_gate_result_1, output_gate_result_1,
        forget_gate_last_candidate_mul_cache_1,
        input_gate_tanh_gate_mul_cache_1, tanh_new_candidate_cache_1,
        /* new_candidate = */ lstm_candidate2_1,
        /* lstm_output_state = */ lstm_state2_1);

    lstm<LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>(
        forget_gate_kernel_last_state_2, forget_gate_kernel_input_state_2,
        forget_gate_bias_2, input_gate_kernel_last_state_2,
        input_gate_kernel_input_state_2, input_gate_bias_2,
        tanh_gate_kernel_last_state_2, tanh_gate_kernel_input_state_2,
        tanh_gate_bias_2, output_gate_kernel_last_state_2,
        output_gate_kernel_input_state_2, output_gate_bias_2,
        /* lstm_last_state = */ lstm_state1_2,
        /* lstm_input_state */ lstm_state2_1,
        /* lstm_last_candidate = */ lstm_candidate1_2, forget_gate_result_2,
        input_gate_result_2, tanh_gate_result_2, output_gate_result_2,
        forget_gate_last_candidate_mul_cache_2,
        input_gate_tanh_gate_mul_cache_2, tanh_new_candidate_cache_2,
        /* new_candidate = */ lstm_candidate2_2,
        /* lstm_output_state */ lstm_state2_2);

    fc(lstm_state2_2, fc_kernel, fc_bias, fc_output_feature_map);

    results[compute_time * 2] = argmax<FDATA_T, IDATA_T>(fc_output_feature_map);

    // pong
    lstm<LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>(
        forget_gate_kernel_last_state_1, forget_gate_kernel_input_state_1,
        forget_gate_bias_1, input_gate_kernel_last_state_1,
        input_gate_kernel_input_state_1, input_gate_bias_1,
        tanh_gate_kernel_last_state_1, tanh_gate_kernel_input_state_1,
        tanh_gate_bias_1, output_gate_kernel_last_state_1,
        output_gate_kernel_input_state_1, output_gate_bias_1,
        /* lstm_last_state = */ lstm_state2_1,
        /* lstm_input_state = */ lstm_input_state_1 +
        LSTM_INPUT_SIZE_1 * (compute_time * 2 + 1),
        /* lstm_last_candidate = */ lstm_candidate2_1, forget_gate_result_1,
        input_gate_result_1, tanh_gate_result_1, output_gate_result_1,
        forget_gate_last_candidate_mul_cache_1,
        input_gate_tanh_gate_mul_cache_1, tanh_new_candidate_cache_1,
        /* new_candidate = */ lstm_candidate1_1,
        /* lstm_output_state = */ lstm_state1_1);

    lstm<LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>(
        forget_gate_kernel_last_state_2, forget_gate_kernel_input_state_2,
        forget_gate_bias_2, input_gate_kernel_last_state_2,
        input_gate_kernel_input_state_2, input_gate_bias_2,
        tanh_gate_kernel_last_state_2, tanh_gate_kernel_input_state_2,
        tanh_gate_bias_2, output_gate_kernel_last_state_2,
        output_gate_kernel_input_state_2, output_gate_bias_2,
        /* lstm_last_state = */ lstm_state2_2,
        /* lstm_input_state */ lstm_state1_1,
        /* lstm_last_candidate = */ lstm_candidate2_2, forget_gate_result_2,
        input_gate_result_2, tanh_gate_result_2, output_gate_result_2,
        forget_gate_last_candidate_mul_cache_2,
        input_gate_tanh_gate_mul_cache_2, tanh_new_candidate_cache_2,
        /* new_candidate = */ lstm_candidate1_2,
        /* lstm_output_state */ lstm_state1_2);

    fc(lstm_state1_2, fc_kernel, fc_bias, fc_output_feature_map);

    results[compute_time*2+1] = argmax<FDATA_T, IDATA_T>(fc_output_feature_map);
  }
}

////////////////////               LSTM                     ////////////////////

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

////////////////////           Fully-Connected              ////////////////////

void fc(const FDATA_T fc_input_feature_map[LSTM_STATE_SIZE_2],
        const FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        const FDATA_T fc_bias[FC_OUTPUT_SIZE],
        FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE]) {
  // input:
  // fc_input_feature_map: a vector with a size of FC_INPUT_SIZE
  // fc_kernel: FC_OUTPUT_SIZE x FC_INPUT_SIZE, notice that this kernel is
  //  TRANSPOSED.
  // fc_bias: FC_OUTPUT_SIZE
  // output:
  // fc_output_feature_map: a vector with a size of FC_OUTPUT_SIZE

  for (IDATA_T result_idx = 0; result_idx < FC_OUTPUT_SIZE; result_idx++) {

    // initialization
    fc_output_feature_map[result_idx] = 0;

    // matrix multiplication
    for (IDATA_T sum_idx = 0; sum_idx < FC_INPUT_SIZE; sum_idx++) {

      IDATA_T fc_kernel_idx = result_idx * FC_INPUT_SIZE + sum_idx;

      fc_output_feature_map[result_idx] += fc_kernel[fc_kernel_idx] *
                                           fc_input_feature_map[sum_idx];
    }

    // add bias
    fc_output_feature_map[result_idx] += fc_bias[result_idx];
  }
}

////////////////////              Activations               ////////////////////

template <>
IDATA_T argmax(FDATA_T* input_array) {
  // for fixed length array (FC_OUTPUT_SIZE)

  // initialization
  IDATA_T max_idx = 0;
  FDATA_T max_val = input_array[0];

  // find max
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++) {
    if (input_array[i] > max_val) {
      max_val = input_array[i];
      max_idx = i;
    }
  }

  return max_idx;
}

template <const int lstm_state_size>
void tanh(FDATA_T* input_feature_map, FDATA_T* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    output_feature_map[result_idx] =
        FDATA_T(tanh(TOFLOAT(input_feature_map[result_idx])));
  }
}

template <const int lstm_state_size>
void sigmoid(FDATA_T* input_feature_map, FDATA_T* output_feature_map) {
  // for fixed length array (LSTM_SIZE), input and output can be the SAME array

  for (LDATA_T result_idx = 0; result_idx < lstm_state_size; result_idx++) {
    output_feature_map[result_idx] =
        1 / (1 + FDATA_T(exp(TOFLOAT(-input_feature_map[result_idx]))));
  }
}

// // instantiation
// template void tanh<LSTM_STATE_SIZE_1>(
    // FDATA_T* input_feature_map, FDATA_T* output_feature_map);

// template void sigmoid<LSTM_STATE_SIZE_1>(
    // FDATA_T* input_feature_map, FDATA_T* output_feature_map);

////////////////////                 Utils                  ////////////////////

template<>
void zero_init(FDATA_T* input_array, LDATA_T array_length)
{
    for(LDATA_T idx = 0; idx < array_length; idx++)
        input_array[idx] = 0;
}

