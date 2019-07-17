#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "constants.h"
#include "types.h"
#include "utils.h"
#include "wrapper.h"

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

  // Layer 1, a super large array, contain the inputs of all steps
  FDATA_T* lstm_input_state_1 =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * LSTM_INPUT_SIZE_1 * COMPUTE_TIME);

  // fc weights
  FDATA_T* fc_kernel =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE * FC_INPUT_SIZE);
  FDATA_T* fc_bias =
      (FDATA_T*) MALLOC(sizeof(FDATA_T) * FC_OUTPUT_SIZE);

  IDATA_T* results = (IDATA_T*) MALLOC(sizeof(IDATA_T) * COMPUTE_TIME);

  printf ("INFO: Start Inferencen\n");

  wrapper(forget_gate_kernel_last_state_1, forget_gate_kernel_input_state_1,
          forget_gate_bias_1, input_gate_kernel_last_state_1,
          input_gate_kernel_input_state_1, input_gate_bias_1,
          tanh_gate_kernel_last_state_1, tanh_gate_kernel_input_state_1,
          tanh_gate_bias_1, output_gate_kernel_last_state_1,
          output_gate_kernel_input_state_1, output_gate_bias_1,
          forget_gate_kernel_last_state_2, forget_gate_kernel_input_state_2,
          forget_gate_bias_2, input_gate_kernel_last_state_2,
          input_gate_kernel_input_state_2, input_gate_bias_2,
          tanh_gate_kernel_last_state_2, tanh_gate_kernel_input_state_2,
          tanh_gate_bias_2, output_gate_kernel_last_state_2,
          output_gate_kernel_input_state_2, output_gate_bias_2, fc_kernel,
          fc_bias, lstm_input_state_1, results);

#ifdef PROFILING
  struct timespec start, finish;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

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

  // fc weights
  MFREE(fc_kernel);
  MFREE(fc_bias);

  MFREE(results);

  return 0;
}
