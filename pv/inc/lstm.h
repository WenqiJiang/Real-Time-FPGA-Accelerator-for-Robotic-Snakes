#pragma once

// the complete LSTM cell 
template <typename FT>
void lstm(FT* forget_gate_kernel_last_state,
          FT* forget_gate_kernel_input_state, FT* forget_gate_bias,
          FT* input_gate_kernel_last_state,
          FT* input_gate_kernel_input_state, FT* input_gate_bias,
          FT* tanh_gate_kernel_last_state, 
          FT* tanh_gate_kernel_input_state, FT* tanh_gate_bias,
          FT* output_gate_kernel_last_state,
          FT* output_gate_kernel_input_state, FT* output_gate_bias,
          FT* last_state, FT* input_state, FT* last_candidate,
          FT* forget_gate_last_candidate_mul_cache, 
          FT* input_gate_tanh_gate_mul_cache,
          FT* tanh_new_candidate_cache,
          FT* new_candidate, FT* output_state);

// forget gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT>
void forget_gate(FT* forget_gate_kernel_last_state,
                 FT* forget_gate_kernel_input_state, FT* forget_gate_bias,
                 FT* last_state, FT* input_state, FT* forget_gate_result);

// input gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT>
void input_gate(FT* input_gate_kernel_last_state,
                FT* input_gate_kernel_input_state, FT* input_gate_bias,
                FT* last_state, FT* input_state, FT* input_gate_result);

// tanh gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT>
void tanh_gate(FT* tanh_gate_kernel_last_state,
               FT* tanh_gate_kernel_input_state, FT* tanh_gate_bias,
               FT* last_state, FT* input_state, FT* tanh_gate_result);

// output gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT>
void output_gate(FT* output_gate_kernel_last_state,
                 FT* output_gate_kernel_input_state, FT* output_gate_bias,
                 FT* last_state, FT* input_state, FT* output_gate_result);

// the computation template for forget, input, tanh, and output gates
template <typename FT>
void gate_template(FT* kernel_last_state, FT* kernel_input_state, FT* bias,
                   FT* last_state, FT* input_state, FT* result);

// this elementwise multiplication has a fixed size, i.e. LSTM_STATE_SIZE
template <typename FT>
void elementwise_mul(FT* input_vector1, FT* input_vector2, FT* output_vector);
