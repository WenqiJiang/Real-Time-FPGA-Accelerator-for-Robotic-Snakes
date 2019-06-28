#include <cstdio>
#include <cstdlib>

#include "activations.h"
#include "constants.h"
#include "fc.h"
#include "lstm.h"
#include "types.h"
#include "utils.h"

int main (int argc, char* argv[]) {

  // input states
  FDATA_T* lstm_input_state = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_INPUT_SIZE);
  FDATA_T* lstm_last_state = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* lstm_last_candidate = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);

  // lstm weights
  FDATA_T* forget_gate_kernel_last_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_STATE_SIZE);
  FDATA_T* forget_gate_kernel_input_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_INPUT_SIZE);
  FDATA_T* forget_gate_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* input_gate_kernel_last_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_STATE_SIZE);
  FDATA_T* input_gate_kernel_input_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_INPUT_SIZE);
  FDATA_T* input_gate_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* tanh_gate_kernel_last_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_STATE_SIZE);
  FDATA_T* tanh_gate_kernel_input_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_INPUT_SIZE);
  FDATA_T* tanh_gate_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* output_gate_kernel_last_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_STATE_SIZE);
  FDATA_T* output_gate_kernel_input_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE * LSTM_INPUT_SIZE);
  FDATA_T* output_gate_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);

  // lstm caches (reuse this case in lstm cells, avoid malloc repeatly
  FDATA_T* forget_gate_result =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* input_gate_result =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* tanh_gate_result =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* output_gate_result =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* forget_gate_last_candidate_mul_cache = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* input_gate_tanh_gate_mul_cache =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* tanh_new_candidate_cache =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);

  // lstm outputs
  FDATA_T* new_candidate =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);
  FDATA_T* lstm_output_state =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE);

  // fc weights
  FDATA_T* fc_kernel = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE * FC_INPUT_SIZE);
  FDATA_T* fc_bias = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);

  // fc output
  FDATA_T* fc_output_feature_map = 
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);

  // TODO: INLDATA_T
  //
  //
  //
  //

  // compute
  lstm<FDATA_T>(
      forget_gate_kernel_last_state, forget_gate_kernel_input_state,
      forget_gate_bias, input_gate_kernel_last_state,
      input_gate_kernel_input_state, input_gate_bias,
      tanh_gate_kernel_last_state, tanh_gate_kernel_input_state,
      tanh_gate_bias, output_gate_kernel_last_state,
      output_gate_kernel_input_state, output_gate_bias, lstm_last_state,
      lstm_input_state, lstm_last_candidate, forget_gate_result,
      input_gate_result, tanh_gate_result, output_gate_result, 
      forget_gate_last_candidate_mul_cache, input_gate_tanh_gate_mul_cache,
      tanh_new_candidate_cache, new_candidate, lstm_output_state);

  fc<FDATA_T>(lstm_output_state, fc_kernel, fc_bias, fc_output_feature_map);
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++) 
    printf("%f\t", fc_output_feature_map[i]);
  printf("\n");

  LDATA_T result = argmax<FDATA_T, LDATA_T>(fc_output_feature_map);
  printf("INFO: result class -> %d\n", result);

  // lstm states
  MFREE(lstm_input_state);
  MFREE(lstm_last_state);
  MFREE(lstm_last_candidate);

  // lstm weights
  MFREE(forget_gate_kernel_last_state);
  MFREE(forget_gate_kernel_input_state);
  MFREE(forget_gate_bias);
  MFREE(input_gate_kernel_last_state);
  MFREE(input_gate_kernel_input_state);
  MFREE(input_gate_bias);
  MFREE(tanh_gate_kernel_last_state);
  MFREE(tanh_gate_kernel_input_state);
  MFREE(tanh_gate_bias);
  MFREE(output_gate_kernel_last_state);
  MFREE(output_gate_kernel_input_state);
  MFREE(output_gate_bias);

  // lstm caches
  MFREE(forget_gate_result);
  MFREE(input_gate_result);
  MFREE(tanh_gate_result);
  MFREE(output_gate_result);
  MFREE(forget_gate_last_candidate_mul_cache);
  MFREE(input_gate_tanh_gate_mul_cache);
  MFREE(tanh_new_candidate_cache);
  
  // lstm outputs
  MFREE(new_candidate);
  MFREE(lstm_output_state);

  // fc weights
  MFREE(fc_kernel);
  MFREE(fc_bias);

  // fc output
  MFREE(fc_output_feature_map);

  return 0;
}
