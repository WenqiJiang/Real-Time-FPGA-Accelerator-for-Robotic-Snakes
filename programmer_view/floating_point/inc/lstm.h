#pragma once

#include "constants.h"
#include "types.h"

// the complete LSTM cell
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
