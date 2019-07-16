#pragma once

#include "constants.h"
#include "types.h"

void wrapper(const FDATA_T forget_gate_kernel_last_state_1
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
