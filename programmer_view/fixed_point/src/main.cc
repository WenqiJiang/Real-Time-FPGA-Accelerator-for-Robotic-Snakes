#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "activations.h"
#include "constants.h"
#include "fc.h"
#include "lstm.h"
#include "types.h"
#include "utils.h"

int main (int argc, char* argv[]) {


  // LSTM Layer 1 weights
  FDATA_T* forget_gate_kernel_last_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  FDATA_T* forget_gate_kernel_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  FDATA_T* forget_gate_bias_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* input_gate_kernel_last_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  FDATA_T* input_gate_kernel_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  FDATA_T* input_gate_bias_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* tanh_gate_kernel_last_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  FDATA_T* tanh_gate_kernel_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  FDATA_T* tanh_gate_bias_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* output_gate_kernel_last_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  FDATA_T* output_gate_kernel_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  FDATA_T* output_gate_bias_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);

  // LSTM Layer 2 weights
  FDATA_T* forget_gate_kernel_last_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  FDATA_T* forget_gate_kernel_input_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  FDATA_T* forget_gate_bias_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* input_gate_kernel_last_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  FDATA_T* input_gate_kernel_input_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  FDATA_T* input_gate_bias_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* tanh_gate_kernel_last_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  FDATA_T* tanh_gate_kernel_input_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  FDATA_T* tanh_gate_bias_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* output_gate_kernel_last_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  FDATA_T* output_gate_kernel_input_state_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T)* LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  FDATA_T* output_gate_bias_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);

  // lstm caches (reuse this case in lstm cells, avoid malloc repeatly
  // Layer 1
  FDATA_T* forget_gate_result_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* input_gate_result_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* tanh_gate_result_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* output_gate_result_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* forget_gate_last_candidate_mul_cache_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* input_gate_tanh_gate_mul_cache_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* tanh_new_candidate_cache_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);

  // Layer 2
  FDATA_T* forget_gate_result_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* input_gate_result_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* tanh_gate_result_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* output_gate_result_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* forget_gate_last_candidate_mul_cache_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* input_gate_tanh_gate_mul_cache_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* tanh_new_candidate_cache_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);

  // LSTM states
  // Layer 1, a super large array, contain the inputs of all steps
  FDATA_T* lstm_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_INPUT_SIZE_1 * COMPUTE_TIME);
  // ping-pong
  FDATA_T* lstm_state1_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* lstm_state2_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* lstm_candidate1_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);
  FDATA_T* lstm_candidate2_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_1);

  // Layer 2
  FDATA_T* lstm_state1_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* lstm_state2_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* lstm_candidate1_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);
  FDATA_T* lstm_candidate2_2 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_STATE_SIZE_2);

  // fc weights
  FDATA_T* fc_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE * FC_INPUT_SIZE);
  FDATA_T* fc_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);

  // fc output
  FDATA_T* fc_output_feature_map =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);
  IDATA_T* results = (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME);

  // TODO: INLDATA_T
  //
  //
  //
  //

  printf ("INFO: Start Inferencen\n");

#ifdef PROFILING
  struct timespec start, finish;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

  // compute
  // Layer 1

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

    fc<FDATA_T>(lstm_state2_2, fc_kernel, fc_bias, fc_output_feature_map);

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

    fc<FDATA_T>(lstm_state1_2, fc_kernel, fc_bias, fc_output_feature_map);

    results[compute_time*2+1] = argmax<FDATA_T, IDATA_T>(fc_output_feature_map);
  }

#ifdef PROFILING
  clock_gettime(CLOCK_REALTIME, &finish);

  long seconds = finish.tv_sec - start.tv_sec;
  long ns = finish.tv_nsec - start.tv_nsec;

  printf("seconds: %ld\n", seconds);
  printf("nanoseconds: %ld\n", ns);
  printf("total seconds: %e\n", (double)seconds + (double)ns/(double)1000000000);
#endif

  printf("INFO: End Inference\n");
#ifdef VERBOSE
  for (LDATA_T i = 0; i < FC_OUTPUT_SIZE; i++)
    printf("%f\t", fc_output_feature_map[i]);
  printf("\n");
#endif

#define PRINT_RESULT
#ifdef PRINT_RESULT
  for (LDATA_T i = 0; i < COMPUTE_TIME; i++) {
    printf("%d\t", results[i]);
  }
  printf("\n");
#endif

  // lstm states
  MFREE(lstm_input_state_1);
  MFREE(lstm_state1_1);
  MFREE(lstm_state2_1);
  MFREE(lstm_candidate1_1);
  MFREE(lstm_candidate2_1);
  MFREE(lstm_state1_2);
  MFREE(lstm_state2_2);
  MFREE(lstm_candidate1_2);
  MFREE(lstm_candidate2_2);

  // lstm weights
  MFREE(forget_gate_kernel_last_state_1);
  MFREE(forget_gate_kernel_input_state_1);
  MFREE(forget_gate_bias_1);
  MFREE(input_gate_kernel_last_state_1);
  MFREE(input_gate_kernel_input_state_1);
  MFREE(input_gate_bias_1);
  MFREE(tanh_gate_kernel_last_state_1);
  MFREE(tanh_gate_kernel_input_state_1);
  MFREE(tanh_gate_bias_1);
  MFREE(output_gate_kernel_last_state_1);
  MFREE(output_gate_kernel_input_state_1);
  MFREE(output_gate_bias_1);
  MFREE(forget_gate_kernel_last_state_2);
  MFREE(forget_gate_kernel_input_state_2);
  MFREE(forget_gate_bias_2);
  MFREE(input_gate_kernel_last_state_2);
  MFREE(input_gate_kernel_input_state_2);
  MFREE(input_gate_bias_2);
  MFREE(tanh_gate_kernel_last_state_2);
  MFREE(tanh_gate_kernel_input_state_2);
  MFREE(tanh_gate_bias_2);
  MFREE(output_gate_kernel_last_state_2);
  MFREE(output_gate_kernel_input_state_2);
  MFREE(output_gate_bias_2);

  // lstm caches
  MFREE(forget_gate_result_1);
  MFREE(input_gate_result_1);
  MFREE(tanh_gate_result_1);
  MFREE(output_gate_result_1);
  MFREE(forget_gate_last_candidate_mul_cache_1);
  MFREE(input_gate_tanh_gate_mul_cache_1);
  MFREE(tanh_new_candidate_cache_1);
  MFREE(forget_gate_result_2);
  MFREE(input_gate_result_2);
  MFREE(tanh_gate_result_2);
  MFREE(output_gate_result_2);
  MFREE(forget_gate_last_candidate_mul_cache_2);
  MFREE(input_gate_tanh_gate_mul_cache_2);
  MFREE(tanh_new_candidate_cache_2);

  // fc weights
  MFREE(fc_kernel);
  MFREE(fc_bias);
  MFREE(results);

  // fc output
  MFREE(fc_output_feature_map);

  return 0;
}
