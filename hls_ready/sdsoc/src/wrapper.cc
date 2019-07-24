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

#pragma SDS data zero_copy(results[0: COMPUTE_TIME * FC_OUTPUT_SIZE])

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
    FDATA_T results[COMPUTE_TIME * FC_OUTPUT_SIZE]) {

////////////////////    INIT EVERYTHING IN BRAM / REG       ////////////////////

  FDATA_T forget_gate_kernel_last_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1];
  FDATA_T forget_gate_kernel_input_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1];
  FDATA_T forget_gate_bias_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T input_gate_kernel_last_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1];
  FDATA_T input_gate_kernel_input_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1];
  FDATA_T input_gate_bias_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T tanh_gate_kernel_last_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1];
  FDATA_T tanh_gate_kernel_input_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1];
  FDATA_T tanh_gate_bias_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T output_gate_kernel_last_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1];
  FDATA_T output_gate_kernel_input_state_1_BRAM
      [LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1];
  FDATA_T output_gate_bias_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T forget_gate_kernel_last_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2];
  FDATA_T forget_gate_kernel_input_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2];
  FDATA_T forget_gate_bias_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T input_gate_kernel_last_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2];
  FDATA_T input_gate_kernel_input_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2];
  FDATA_T input_gate_bias_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T tanh_gate_kernel_last_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2];
  FDATA_T tanh_gate_kernel_input_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2];
  FDATA_T tanh_gate_bias_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T output_gate_kernel_last_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2];
  FDATA_T output_gate_kernel_input_state_2_BRAM
      [LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2];
  FDATA_T output_gate_bias_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T fc_kernel_BRAM[FC_OUTPUT_SIZE * FC_INPUT_SIZE];
  FDATA_T fc_bias_BRAM[FC_OUTPUT_SIZE];

#pragma HLS array_partition variable=forget_gate_kernel_last_state_1_BRAM \
    complete
#pragma HLS array_partition variable=forget_gate_kernel_input_state_1_BRAM \
    complete
#pragma HLS array_partition variable=forget_gate_bias_1_BRAM complete
#pragma HLS array_partition variable=input_gate_kernel_last_state_1_BRAM \
    complete
#pragma HLS array_partition variable=input_gate_kernel_input_state_1_BRAM \
    complete
#pragma HLS array_partition variable=input_gate_bias_1_BRAM complete
#pragma HLS array_partition variable=tanh_gate_kernel_last_state_1_BRAM complete
#pragma HLS array_partition variable=tanh_gate_kernel_input_state_1_BRAM \
    complete
#pragma HLS array_partition variable=tanh_gate_bias_1_BRAM complete
#pragma HLS array_partition variable=output_gate_kernel_last_state_1_BRAM \
    complete
#pragma HLS array_partition variable=output_gate_kernel_input_state_1_BRAM \
    complete
#pragma HLS array_partition variable=output_gate_bias_1_BRAM complete
#pragma HLS array_partition variable=forget_gate_kernel_last_state_2_BRAM \
    complete
#pragma HLS array_partition variable=forget_gate_kernel_input_state_2_BRAM \
    complete
#pragma HLS array_partition variable=forget_gate_bias_2_BRAM complete
#pragma HLS array_partition variable=input_gate_kernel_last_state_2_BRAM \
    complete
#pragma HLS array_partition variable=input_gate_kernel_input_state_2_BRAM \
    complete
#pragma HLS array_partition variable=input_gate_bias_2_BRAM complete
#pragma HLS array_partition variable=tanh_gate_kernel_last_state_2_BRAM \
    complete
#pragma HLS array_partition variable=tanh_gate_kernel_input_state_2_BRAM \
    complete
#pragma HLS array_partition variable=tanh_gate_bias_2_BRAM complete
#pragma HLS array_partition variable=output_gate_kernel_last_state_2_BRAM \
    complete
#pragma HLS array_partition variable=output_gate_kernel_input_state_2_BRAM \
    complete
#pragma HLS array_partition variable=output_gate_bias_2_BRAM complete
#pragma HLS array_partition variable=fc_kernel_BRAM complete
#pragma HLS array_partition variable=fc_bias_BRAM complete

  copy_array<LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1>(
      forget_gate_kernel_last_state_1_BRAM, forget_gate_kernel_last_state_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1>(
    forget_gate_kernel_input_state_1_BRAM, forget_gate_kernel_input_state_1);
  copy_array<LSTM_STATE_SIZE_1>(forget_gate_bias_1_BRAM, forget_gate_bias_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1>(
    input_gate_kernel_last_state_1_BRAM, input_gate_kernel_last_state_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1>(
    input_gate_kernel_input_state_1_BRAM, input_gate_kernel_input_state_1);
  copy_array<LSTM_STATE_SIZE_1>(input_gate_bias_1_BRAM, input_gate_bias_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1>(
    tanh_gate_kernel_last_state_1_BRAM, tanh_gate_kernel_last_state_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1>(
    tanh_gate_kernel_input_state_1_BRAM, tanh_gate_kernel_input_state_1);
  copy_array<LSTM_STATE_SIZE_1>(tanh_gate_bias_1_BRAM, tanh_gate_bias_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1>(
    output_gate_kernel_last_state_1_BRAM, output_gate_kernel_last_state_1);
  copy_array<LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1>(
    output_gate_kernel_input_state_1_BRAM, output_gate_kernel_input_state_1);
  copy_array<LSTM_STATE_SIZE_1>(output_gate_bias_1_BRAM, output_gate_bias_1);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2>(
    forget_gate_kernel_last_state_2_BRAM, forget_gate_kernel_last_state_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2>(
    forget_gate_kernel_input_state_2_BRAM, forget_gate_kernel_input_state_2);
  copy_array<LSTM_STATE_SIZE_2>(forget_gate_bias_2_BRAM, forget_gate_bias_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2>(
    input_gate_kernel_last_state_2_BRAM, input_gate_kernel_last_state_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2>(
    input_gate_kernel_input_state_2_BRAM, input_gate_kernel_input_state_2);
  copy_array<LSTM_STATE_SIZE_2>(input_gate_bias_2_BRAM, input_gate_bias_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2>(
    tanh_gate_kernel_last_state_2_BRAM, tanh_gate_kernel_last_state_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2>(
    tanh_gate_kernel_input_state_2_BRAM, tanh_gate_kernel_input_state_2);
  copy_array<LSTM_STATE_SIZE_2>(tanh_gate_bias_2_BRAM, tanh_gate_bias_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2>(
    output_gate_kernel_last_state_2_BRAM, output_gate_kernel_last_state_2);
  copy_array<LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2>(
    output_gate_kernel_input_state_2_BRAM, output_gate_kernel_input_state_2);
  copy_array<LSTM_STATE_SIZE_2>(output_gate_bias_2_BRAM, output_gate_bias_2);
  copy_array<FC_OUTPUT_SIZE * FC_INPUT_SIZE>(fc_kernel_BRAM, fc_kernel);
  copy_array<FC_OUTPUT_SIZE>(fc_bias_BRAM, fc_bias);

  // lstm caches (reuse this case in lstm cells, avoid malloc repeatly
  // Layer 1
  FDATA_T forget_gate_result_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T input_gate_result_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T tanh_gate_result_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T output_gate_result_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T forget_gate_last_candidate_mul_cache_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T input_gate_tanh_gate_mul_cache_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T tanh_new_candidate_cache_1_BRAM[LSTM_STATE_SIZE_1];

  // Layer 2
  FDATA_T forget_gate_result_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T input_gate_result_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T tanh_gate_result_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T output_gate_result_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T forget_gate_last_candidate_mul_cache_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T input_gate_tanh_gate_mul_cache_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T tanh_new_candidate_cache_2_BRAM[LSTM_STATE_SIZE_2];

  // LSTM states
  // ping-pong
  FDATA_T lstm_state1_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T lstm_state2_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T lstm_candidate1_1_BRAM[LSTM_STATE_SIZE_1];
  FDATA_T lstm_candidate2_1_BRAM[LSTM_STATE_SIZE_1];

  // Layer 2
  FDATA_T lstm_state1_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T lstm_state2_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T lstm_candidate1_2_BRAM[LSTM_STATE_SIZE_2];
  FDATA_T lstm_candidate2_2_BRAM[LSTM_STATE_SIZE_2];

  // fc output
  FDATA_T fc_output_feature_map_BRAM[FC_OUTPUT_SIZE];

#pragma HLS array_partition variable=forget_gate_result_1_BRAM complete
#pragma HLS array_partition variable=input_gate_result_1_BRAM complete
#pragma HLS array_partition variable=tanh_gate_result_1_BRAM complete
#pragma HLS array_partition variable=output_gate_result_1_BRAM complete
#pragma HLS array_partition \
    variable=forget_gate_last_candidate_mul_cache_1_BRAM complete
#pragma HLS array_partition \
    variable=input_gate_tanh_gate_mul_cache_1_BRAM complete
#pragma HLS array_partition variable=tanh_new_candidate_cache_1_BRAM complete
#pragma HLS array_partition variable=forget_gate_result_2_BRAM complete
#pragma HLS array_partition variable=input_gate_result_2_BRAM complete
#pragma HLS array_partition variable=tanh_gate_result_2_BRAM complete
#pragma HLS array_partition variable=output_gate_result_2_BRAM complete
#pragma HLS array_partition \
    variable=forget_gate_last_candidate_mul_cache_2_BRAM complete
#pragma HLS array_partition \
    variable=input_gate_tanh_gate_mul_cache_2_BRAM complete
#pragma HLS array_partition variable=tanh_new_candidate_cache_2_BRAM complete
#pragma HLS array_partition variable=lstm_state1_1_BRAM complete
#pragma HLS array_partition variable=lstm_state2_1_BRAM complete
#pragma HLS array_partition variable=lstm_candidate1_1_BRAM complete
#pragma HLS array_partition variable=lstm_candidate2_1_BRAM complete
#pragma HLS array_partition variable=lstm_state1_2_BRAM complete
#pragma HLS array_partition variable=lstm_state2_2_BRAM complete
#pragma HLS array_partition variable=lstm_candidate1_2_BRAM complete
#pragma HLS array_partition variable=lstm_candidate2_2_BRAM complete
#pragma HLS array_partition variable=fc_output_feature_map_BRAM complete

////////////////////            START COMPUTING             ////////////////////

  // ping-pong
  for (LDATA_T compute_time = 0; compute_time < COMPUTE_TIME / 2;
       compute_time++) {

    // ping
    lstm<LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>(
        forget_gate_kernel_last_state_1_BRAM,
        forget_gate_kernel_input_state_1_BRAM,
        forget_gate_bias_1_BRAM, input_gate_kernel_last_state_1_BRAM,
        input_gate_kernel_input_state_1_BRAM, input_gate_bias_1_BRAM,
        tanh_gate_kernel_last_state_1_BRAM, tanh_gate_kernel_input_state_1_BRAM,
        tanh_gate_bias_1_BRAM, output_gate_kernel_last_state_1_BRAM,
        output_gate_kernel_input_state_1_BRAM, output_gate_bias_1_BRAM,
        /* lstm_last_state = */ lstm_state1_1_BRAM,
        /* lstm_input_state = */ lstm_input_state_1 +
        LSTM_INPUT_SIZE_1 * compute_time * 2,
        /* lstm_last_candidate = */ lstm_candidate1_1_BRAM,
        forget_gate_result_1_BRAM,
        input_gate_result_1_BRAM, tanh_gate_result_1_BRAM,
        output_gate_result_1_BRAM, forget_gate_last_candidate_mul_cache_1_BRAM,
        input_gate_tanh_gate_mul_cache_1_BRAM, tanh_new_candidate_cache_1_BRAM,
        /* new_candidate = */ lstm_candidate2_1_BRAM,
        /* lstm_output_state = */ lstm_state2_1_BRAM);

    lstm<LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>(
        forget_gate_kernel_last_state_2_BRAM,
        forget_gate_kernel_input_state_2_BRAM,
        forget_gate_bias_2_BRAM, input_gate_kernel_last_state_2_BRAM,
        input_gate_kernel_input_state_2_BRAM, input_gate_bias_2_BRAM,
        tanh_gate_kernel_last_state_2_BRAM, tanh_gate_kernel_input_state_2_BRAM,
        tanh_gate_bias_2_BRAM, output_gate_kernel_last_state_2_BRAM,
        output_gate_kernel_input_state_2_BRAM, output_gate_bias_2_BRAM,
        /* lstm_last_state = */ lstm_state1_2_BRAM,
        /* lstm_input_state */ lstm_state2_1_BRAM,
        /* lstm_last_candidate = */ lstm_candidate1_2_BRAM,
        forget_gate_result_2_BRAM, input_gate_result_2_BRAM,
        tanh_gate_result_2_BRAM, output_gate_result_2_BRAM,
        forget_gate_last_candidate_mul_cache_2_BRAM,
        input_gate_tanh_gate_mul_cache_2_BRAM, tanh_new_candidate_cache_2_BRAM,
        /* new_candidate = */ lstm_candidate2_2_BRAM,
        /* lstm_output_state */ lstm_state2_2_BRAM);

    fc(lstm_state2_2_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
       fc_output_feature_map_BRAM);

    copy_array<FC_OUTPUT_SIZE> (results + 2 * compute_time * FC_OUTPUT_SIZE,
                                fc_output_feature_map_BRAM);

    // pong
    lstm<LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>(
        forget_gate_kernel_last_state_1_BRAM,
        forget_gate_kernel_input_state_1_BRAM,
        forget_gate_bias_1_BRAM, input_gate_kernel_last_state_1_BRAM,
        input_gate_kernel_input_state_1_BRAM, input_gate_bias_1_BRAM,
        tanh_gate_kernel_last_state_1_BRAM, tanh_gate_kernel_input_state_1_BRAM,
        tanh_gate_bias_1_BRAM, output_gate_kernel_last_state_1_BRAM,
        output_gate_kernel_input_state_1_BRAM, output_gate_bias_1_BRAM,
        /* lstm_last_state = */ lstm_state2_1_BRAM,
        /* lstm_input_state = */ lstm_input_state_1 +
        LSTM_INPUT_SIZE_1 * (compute_time * 2 + 1),
        /* lstm_last_candidate = */ lstm_candidate2_1_BRAM,
        forget_gate_result_1_BRAM, input_gate_result_1_BRAM,
        tanh_gate_result_1_BRAM, output_gate_result_1_BRAM,
        forget_gate_last_candidate_mul_cache_1_BRAM,
        input_gate_tanh_gate_mul_cache_1_BRAM, tanh_new_candidate_cache_1_BRAM,
        /* new_candidate = */ lstm_candidate1_1_BRAM,
        /* lstm_output_state = */ lstm_state1_1_BRAM);

    lstm<LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>(
        forget_gate_kernel_last_state_2_BRAM,
        forget_gate_kernel_input_state_2_BRAM,
        forget_gate_bias_2_BRAM, input_gate_kernel_last_state_2_BRAM,
        input_gate_kernel_input_state_2_BRAM, input_gate_bias_2_BRAM,
        tanh_gate_kernel_last_state_2_BRAM, tanh_gate_kernel_input_state_2_BRAM,
        tanh_gate_bias_2_BRAM, output_gate_kernel_last_state_2_BRAM,
        output_gate_kernel_input_state_2_BRAM, output_gate_bias_2_BRAM,
        /* lstm_last_state = */ lstm_state2_2_BRAM,
        /* lstm_input_state */ lstm_state1_1_BRAM,
        /* lstm_last_candidate = */ lstm_candidate2_2_BRAM,
        forget_gate_result_2_BRAM, input_gate_result_2_BRAM,
        tanh_gate_result_2_BRAM, output_gate_result_2_BRAM,
        forget_gate_last_candidate_mul_cache_2_BRAM,
        input_gate_tanh_gate_mul_cache_2_BRAM, tanh_new_candidate_cache_2_BRAM,
        /* new_candidate = */ lstm_candidate1_2_BRAM,
        /* lstm_output_state */ lstm_state1_2_BRAM);

    fc(lstm_state1_2_BRAM, fc_kernel_BRAM, fc_bias_BRAM,
       fc_output_feature_map_BRAM);

    copy_array<FC_OUTPUT_SIZE> (results + (2*compute_time + 1) * FC_OUTPUT_SIZE,
                                fc_output_feature_map_BRAM);
  }
}

////////////////////               LSTM                     ////////////////////

template <LSTM_STATE_SIZE_1, LSTM_INPUT_SIZE_1>
void gate_template(
    const FDATA_T kernel_last_state[LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1],
    const FDATA_T kernel_input_state[LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1],
    const FDATA_T bias[LSTM_STATE_SIZE_1],
    const FDATA_T lstm_last_state[LSTM_STATE_SIZE_1],
    const FDATA_T lstm_input_state[LSTM_INPUT_SIZE_1],
    FDATA_T result[LSTM_STATE_SIZE_1]) {

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
	FDATA_T local_reg[LSTM_STATE_SIZE_1][LSTM_STATE_SIZE_1 + LSTM_INPUT_SIZE_1];
#pragma HLS array_partition variable=local_reg[output_state_index] complete

  for (LDATA_T output_state_index = 0; output_state_index < LSTM_STATE_SIZE_1;
      output_state_index++) {
#pragma HLS UNROLL complete

    for (LDATA_T input_state_index = 0; input_state_index < LSTM_INPUT_SIZE_1;
         input_state_index++) {
#pragma HLS RESOURCE variable=local_reg[output_state_index] core=FMul_fulldsp
#pragma HLS UNROLL complete

      local_reg[output_state_index][input_state_index] =
          kernel_input_state[output_state_index * LSTM_INPUT_SIZE_1 +
                             input_state_index] *
          lstm_input_state[input_state_index];
    }

    for (LDATA_T last_state_index = 0; last_state_index < LSTM_STATE_SIZE_1;
         last_state_index++) {
#pragma HLS RESOURCE variable=local_reg[output_state_index] core=FMul_fulldsp
#pragma HLS UNROLL complete

      local_reg[output_state_index][LSTM_INPUT_SIZE_1 + last_state_index] =
          kernel_last_state[output_state_index * LSTM_STATE_SIZE_1 +
                            last_state_index] *
          lstm_last_state[last_state_index];
    }

    ////// HACKING, suppose LSTM_INPUT_SIZE_1 + LSTM_STATE_SIZE_1 = 30 /////

    // prefix sum
    for (LDATA_T i = 0; i < 15; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][15 + i];
    }

    // 15 = 7 * 2 + 1 -> need 8 reg for next iteration
    // the 15'th number will be copy to 8'th reg
    for (LDATA_T i = 0; i < 7; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][7 + i];
    }
    local_reg[output_state_index][7] = local_reg[output_state_index][14];

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 4; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][4 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 2; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][2 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 1; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][1 + i];
    }

    result[output_state_index] = bias[output_state_index] +
                                 local_reg[output_state_index][0];
  }
}

template <LSTM_STATE_SIZE_2, LSTM_INPUT_SIZE_2>
void gate_template(
    const FDATA_T kernel_last_state[LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2],
    const FDATA_T kernel_input_state[LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2],
    const FDATA_T bias[LSTM_STATE_SIZE_2],
    const FDATA_T lstm_last_state[LSTM_STATE_SIZE_2],
    const FDATA_T lstm_input_state[LSTM_INPUT_SIZE_2],
    FDATA_T result[LSTM_STATE_SIZE_2]) {

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
	FDATA_T local_reg[LSTM_STATE_SIZE_2][LSTM_STATE_SIZE_2 + LSTM_INPUT_SIZE_2];
#pragma HLS array_partition variable=local_reg[output_state_index] complete

  for (LDATA_T output_state_index = 0; output_state_index < LSTM_STATE_SIZE_2;
      output_state_index++) {
#pragma HLS UNROLL complete

    for (LDATA_T input_state_index = 0; input_state_index < LSTM_INPUT_SIZE_2;
         input_state_index++) {
#pragma HLS RESOURCE variable=local_reg[output_state_index] core=FMul_fulldsp
#pragma HLS UNROLL complete

      local_reg[output_state_index][input_state_index] =
          kernel_input_state[output_state_index * LSTM_INPUT_SIZE_2 +
                             input_state_index] *
          lstm_input_state[input_state_index];
    }

    for (LDATA_T last_state_index = 0; last_state_index < LSTM_STATE_SIZE_2;
         last_state_index++) {
#pragma HLS RESOURCE variable=local_reg[output_state_index] core=FMul_fulldsp
#pragma HLS UNROLL complete

      local_reg[output_state_index][LSTM_INPUT_SIZE_2 + last_state_index] =
          kernel_last_state[output_state_index * LSTM_STATE_SIZE_2 +
                            last_state_index] *
          lstm_last_state[last_state_index];
    }

    ////// HACKING, suppose LSTM_INPUT_SIZE_2 + LSTM_STATE_SIZE_2 = 32 /////

    // prefix sum
    for (LDATA_T i = 0; i < 16; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][16 + i];
    }

    for (LDATA_T i = 0; i < 8; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][8 + i];
    }

    for (LDATA_T i = 0; i < 4; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][4 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 2; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][2 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 1; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][1 + i];
    }

    result[output_state_index] = bias[output_state_index] +
                                 local_reg[output_state_index][0];
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

  // initialization
	FDATA_T local_reg[FC_OUTPUT_SIZE][FC_INPUT_SIZE];
#pragma HLS array_partition variable=local_reg[output_state_index] complete

  for (LDATA_T output_state_index = 0; output_state_index < FC_OUTPUT_SIZE;
      output_state_index++) {
#pragma HLS UNROLL complete

    for (LDATA_T input_state_index = 0; input_state_index < FC_INPUT_SIZE;
         input_state_index++) {
#pragma HLS RESOURCE variable=local_reg[output_state_index] core=FMul_fulldsp
#pragma HLS UNROLL complete

      local_reg[output_state_index][input_state_index] =
          fc_kernel[output_state_index * FC_INPUT_SIZE + input_state_index] *
          fc_input_feature_map[input_state_index];
    }

    ////// HACKING, suppose FC_INPUT_SIZE = 16 /////

    // prefix sum
    for (LDATA_T i = 0; i < 8; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][8 + i];
    }

    for (LDATA_T i = 0; i < 4; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][4 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 2; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][2 + i];
    }

    // from 8, regular prefix sum
    for (LDATA_T i = 0; i < 1; i++) {
#pragma HLS UNROLL complete
      local_reg[output_state_index][i] = local_reg[output_state_index][i] +
          local_reg[output_state_index][1 + i];
    }

    fc_output_feature_map[output_state_index] =
        fc_bias[output_state_index] + local_reg[output_state_index][0];
  }
}

////////////////////              Activations               ////////////////////

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

template <const int length>
void copy_array(FDATA_T* dst, FDATA_T* src) {

  for (LDATA_T i = 0; i < length; i++) {
#pragma HLS pipeline
    dst[i] = src[i];
  }
}
