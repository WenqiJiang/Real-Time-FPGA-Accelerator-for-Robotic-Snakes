#pragma once

// the complete LSTM cell
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void lstm(const FT* forget_gate_kernel_last_state,
          const FT* forget_gate_kernel_input_state, const FT* forget_gate_bias,
          const FT* input_gate_kernel_last_state,
          const FT* input_gate_kernel_input_state, const FT* input_gate_bias,
          const FT* tanh_gate_kernel_last_state,
          const FT* tanh_gate_kernel_input_state, const FT* tanh_gate_bias,
          const FT* output_gate_kernel_last_state,
          const FT* output_gate_kernel_input_state, const FT* output_gate_bias,
          const FT* lstm_last_state, const FT* lstm_input_state,
          const FT* last_candidate, FT* forget_gate_result,
          FT* input_gate_result, FT* tanh_gate_result,
          FT* output_gate_result, FT* forget_gate_last_candidate_mul_cache,
          FT* input_gate_tanh_gate_mul_cache, FT* tanh_new_candidate_cache,
          FT* new_candidate, FT* lstm_output_state);

// forget gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void forget_gate(const FT* forget_gate_kernel_last_state,
                 const FT* forget_gate_kernel_input_state,
                 const FT* forget_gate_bias, const FT* lstm_last_state,
                 const FT* lstm_input_state, FT* forget_gate_result);

// input gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void input_gate(FT* input_gate_kernel_last_state,
                const FT* input_gate_kernel_input_state,
                const FT* input_gate_bias, const FT* lstm_last_state,
                const FT* lstm_input_state, FT* input_gate_result);

// tanh gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void tanh_gate(const FT* tanh_gate_kernel_last_state,
               const FT* tanh_gate_kernel_input_state, const FT* tanh_gate_bias,
               const FT* lstm_last_state, const FT* lstm_input_state,
               FT* tanh_gate_result);

// output gate is a fixed-length computation, related to LSTM_STATE_SIZE
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void output_gate(const FT* output_gate_kernel_last_state,
                 const FT* output_gate_kernel_input_state,
                 const FT* output_gate_bias, const FT* lstm_last_state,
                 const FT* lstm_input_state, FT* output_gate_result);

// the computation template for forget, input, tanh, and output gates
template <typename FT, const int lstm_state_size, const int lstm_input_size>
void gate_template(const FT* kernel_last_state, const FT* kernel_input_state,
                   const FT* bias, const FT* lstm_last_state,
                   const FT* lstm_input_state, FT* result);

// this elementwise multiplication has a fixed size, i.e. LSTM_STATE_SIZE
template <typename FT, const int lstm_state_size>
void elementwise_mul(const FT* input_vector1, const FT* input_vector2,
                     FT* output_vector);
