#pragma once

#include "constants.h"
#include "types.h"

////////////////////         TOP-LEVEL FUNCTION             ////////////////////

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
    IDATA_T results[COMPUTE_TIME]);

////////////////////                  LSTM                  ////////////////////


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
          FDATA_T lstm_output_state[lstm_state_size]);

// the computation template for forget, input, tanh, and output gates
template <const int lstm_state_size, const int lstm_input_size>
void gate_template(const FDATA_T kernel_last_state
                       [lstm_state_size * lstm_state_size],
                   const FDATA_T kernel_input_state
                       [lstm_state_size * lstm_input_size],
                   const FDATA_T bias[lstm_state_size],
                   const FDATA_T lstm_last_state[lstm_state_size],
                   const FDATA_T lstm_input_state[lstm_input_size],
                   FDATA_T result[lstm_state_size]);

template <const int lstm_state_size, const int lstm_input_size>
void forget_gate(const FDATA_T forget_gate_kernel_last_state
                     [lstm_state_size * lstm_state_size],
                 const FDATA_T forget_gate_kernel_input_state
                     [lstm_state_size * lstm_input_size],
                 const FDATA_T forget_gate_bias[lstm_state_size],
                 const FDATA_T lstm_last_state[lstm_state_size],
                 const FDATA_T lstm_input_state[lstm_input_size],
                 FDATA_T forget_gate_result[lstm_state_size]);

template <const int lstm_state_size, const int lstm_input_size>
void input_gate(const FDATA_T input_gate_kernel_last_state
                    [lstm_state_size * lstm_state_size],
                const FDATA_T input_gate_kernel_input_state
                    [lstm_state_size * lstm_input_size],
                const FDATA_T input_gate_bias[lstm_state_size],
                const FDATA_T lstm_last_state[lstm_state_size],
                const FDATA_T lstm_input_state[lstm_input_size],
                FDATA_T input_gate_result[lstm_state_size]);

template <const int lstm_state_size, const int lstm_input_size>
void tanh_gate(const FDATA_T tanh_gate_kernel_last_state
                   [lstm_state_size * lstm_state_size],
               const FDATA_T tanh_gate_kernel_input_state
                   [lstm_state_size * lstm_input_size],
               const FDATA_T tanh_gate_bias[lstm_state_size],
               const FDATA_T lstm_last_state[lstm_state_size],
               const FDATA_T lstm_input_state[lstm_input_size],
               FDATA_T tanh_gate_result[lstm_state_size]);

template <const int lstm_state_size, const int lstm_input_size>
void output_gate(const FDATA_T output_gate_kernel_last_state
                     [lstm_state_size * lstm_state_size],
                 const FDATA_T output_gate_kernel_input_state
                     [lstm_state_size * lstm_input_size],
                 const FDATA_T output_gate_bias[lstm_state_size],
                 const FDATA_T lstm_last_state[lstm_state_size],
                 const FDATA_T lstm_input_state[lstm_input_size],
                 FDATA_T output_gate_result[lstm_state_size]);

template <const int lstm_state_size>
void elementwise_mul(const FDATA_T input_vector1[lstm_state_size],
                     const FDATA_T input_vector2[lstm_state_size],
                       FDATA_T output_vector[lstm_state_size]);

////////////////////            Fully-Connected             ////////////////////

// fully-connected layer with fixed size of input and output vectors,
//  i.e. FC_INPUT_SIZE and FC_OUTPUT_SIZE separately
void fc(const FDATA_T fc_input_feature_map[LSTM_STATE_SIZE_2],
        const FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        const FDATA_T fc_bias[FC_OUTPUT_SIZE],
        FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE]);

////////////////////              Activations               ////////////////////

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <const int lstm_state_size>
void tanh(FDATA_T* input_feature_map, FDATA_T* output_feature_map);

// for fixed length array (LSTM_SIZE), input and output can be the SAME array
template <const int lstm_state_size>
void sigmoid(FDATA_T* input_feature_map, FDATA_T* output_feature_map);


////////////////////                 Utils                  ////////////////////

// copy array between BRAM and DRAM, thus no array partition, pipeline only
template <const int length>
void copy_array(FDATA_T* dst, FDATA_T* src);

// specification
template <LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1>
void copy_array(FDATA_T* dst, FDATA_T* src);

template <LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1>
void copy_array(FDATA_T* dst, FDATA_T* src);

template <LSTM_STATE_SIZE_1>
void copy_array(FDATA_T* dst, FDATA_T* src);

template <FC_OUTPUT_SIZE * FC_INPUT_SIZE>
void copy_array(FDATA_T* dst, FDATA_T* src);

template <FC_OUTPUT_SIZE>
void copy_array(FDATA_T* dst, FDATA_T* src);

