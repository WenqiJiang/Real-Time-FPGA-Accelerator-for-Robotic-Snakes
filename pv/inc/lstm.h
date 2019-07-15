#pragma once

#include "types.h"

// the complete LSTM cell
template <const int lstm_state_size, const int lstm_input_size>
void lstm(const FDATA_T* forget_gate_kernel_last_state,
          const FDATA_T* forget_gate_kernel_input_state,
          const FDATA_T* forget_gate_bias,
          const FDATA_T* input_gate_kernel_last_state,
          const FDATA_T* input_gate_kernel_input_state,
          const FDATA_T* input_gate_bias,
          const FDATA_T* tanh_gate_kernel_last_state,
          const FDATA_T* tanh_gate_kernel_input_state,
          const FDATA_T* tanh_gate_bias,
          const FDATA_T* output_gate_kernel_last_state,
          const FDATA_T* output_gate_kernel_input_state,
          const FDATA_T* output_gate_bias,
          const FDATA_T* lstm_last_state, const FDATA_T* lstm_input_state,
          const FDATA_T* last_candidate, FDATA_T* forget_gate_result,
          FDATA_T* input_gate_result, FDATA_T* tanh_gate_result,
          FDATA_T* output_gate_result,
          FDATA_T* forget_gate_last_candidate_mul_cache,
          FDATA_T* input_gate_tanh_gate_mul_cache,
          FDATA_T* tanh_new_candidate_cache,
          FDATA_T* new_candidate, FDATA_T* lstm_output_state);

// forget gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <const int lstm_state_size, const int lstm_input_size>
void forget_gate(const FDATA_T* forget_gate_kernel_last_state,
                 const FDATA_T* forget_gate_kernel_input_state,
                 const FDATA_T* forget_gate_bias,
                 const FDATA_T* lstm_last_state,
                 const FDATA_T* lstm_input_state, FDATA_T* forget_gate_result);

// input gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <const int lstm_state_size, const int lstm_input_size>
void input_gate(FDATA_T* input_gate_kernel_last_state,
                const FDATA_T* input_gate_kernel_input_state,
                const FDATA_T* input_gate_bias, const FDATA_T* lstm_last_state,
                const FDATA_T* lstm_input_state, FDATA_T* input_gate_result);

// tanh gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <const int lstm_state_size, const int lstm_input_size>
void tanh_gate(const FDATA_T* tanh_gate_kernel_last_state,
               const FDATA_T* tanh_gate_kernel_input_state,
               const FDATA_T* tanh_gate_bias,
               const FDATA_T* lstm_last_state, const FDATA_T* lstm_input_state,
               FDATA_T* tanh_gate_result);

// output gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <const int lstm_state_size, const int lstm_input_size>
void output_gate(const FDATA_T* output_gate_kernel_last_state,
                 const FDATA_T* output_gate_kernel_input_state,
                 const FDATA_T* output_gate_bias,
                 const FDATA_T* lstm_last_state,
                 const FDATA_T* lstm_input_state, FDATA_T* output_gate_result);

// the computation template for forget, input, tanh, and output gates
template <const int lstm_state_size, const int lstm_input_size>
void gate_template(const FDATA_T* kernel_last_state,
                   const FDATA_T* kernel_input_state,
                   const FDATA_T* bias, const FDATA_T* lstm_last_state,
                   const FDATA_T* lstm_input_state, FDATA_T* result);

// this elementwise multiplication has a fixed size, i.e. LSTM_STATE_SIZE
template <const int lstm_state_size>
void elementwise_mul(const FDATA_T* input_vector1, const FDATA_T* input_vector2,
                     FDATA_T* output_vector);
