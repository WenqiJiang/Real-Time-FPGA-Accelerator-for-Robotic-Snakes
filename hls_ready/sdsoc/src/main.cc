#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#include "configs.h"
#include "constants.h"
#include "types.h"
#include "utils.h"
#include "wrapper.h"

int main (int argc, char* argv[]) {

  printf("INFO: Allocating memory and loading weights...\n");

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

  // Load weights
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_KERNEL_LAST_STATE_1,
      forget_gate_kernel_last_state_1, LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_KERNEL_INPUT_STATE_1,
      forget_gate_kernel_input_state_1 ,LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_BIAS_1,
      forget_gate_bias_1 ,LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_KERNEL_LAST_STATE_1,
      input_gate_kernel_last_state_1 ,LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_KERNEL_INPUT_STATE_1,
      input_gate_kernel_input_state_1 ,LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_BIAS_1,
      input_gate_bias_1 ,LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_KERNEL_LAST_STATE_1,
      tanh_gate_kernel_last_state_1 ,LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_KERNEL_INPUT_STATE_1,
      tanh_gate_kernel_input_state_1 ,LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_BIAS_1,
      tanh_gate_bias_1 ,LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_KERNEL_LAST_STATE_1,
      output_gate_kernel_last_state_1 ,LSTM_STATE_SIZE_1 * LSTM_STATE_SIZE_1);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_KERNEL_INPUT_STATE_1,
      output_gate_kernel_input_state_1 ,LSTM_STATE_SIZE_1 * LSTM_INPUT_SIZE_1);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_BIAS_1,
      output_gate_bias_1 ,LSTM_STATE_SIZE_1);

  // LSTM Layer 2 weights
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_KERNEL_LAST_STATE_2,
      forget_gate_kernel_last_state_2 ,LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_KERNEL_INPUT_STATE_2,
      forget_gate_kernel_input_state_2 ,LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  load_data<FDATA_T, LDATA_T> (FORGET_GATE_BIAS_2,
      forget_gate_bias_2 ,LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_KERNEL_LAST_STATE_2,
      input_gate_kernel_last_state_2 ,LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_KERNEL_INPUT_STATE_2,
      input_gate_kernel_input_state_2 ,LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  load_data<FDATA_T, LDATA_T> (INPUT_GATE_BIAS_2,
      input_gate_bias_2 ,LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_KERNEL_LAST_STATE_2,
      tanh_gate_kernel_last_state_2 ,LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_KERNEL_INPUT_STATE_2,
      tanh_gate_kernel_input_state_2 ,LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  load_data<FDATA_T, LDATA_T> (TANH_GATE_BIAS_2,
      tanh_gate_bias_2 ,LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_KERNEL_LAST_STATE_2,
      output_gate_kernel_last_state_2 ,LSTM_STATE_SIZE_2 * LSTM_STATE_SIZE_2);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_KERNEL_INPUT_STATE_2,
      output_gate_kernel_input_state_2 ,LSTM_STATE_SIZE_2 * LSTM_INPUT_SIZE_2);
  load_data<FDATA_T, LDATA_T> (OUTPUT_GATE_BIAS_2,
      output_gate_bias_2 ,LSTM_STATE_SIZE_2);

  // Layer 1, a super large array, contain the inputs of all steps
  load_data<FDATA_T, LDATA_T> (LSTM_INPUT_STATE_1,
      lstm_input_state_1 ,LSTM_INPUT_SIZE_1 * COMPUTE_TIME);

  // fc weights
  load_data<FDATA_T, LDATA_T> (FC_KERNEL,
      fc_kernel ,FC_OUTPUT_SIZE * FC_INPUT_SIZE);
  load_data<FDATA_T, LDATA_T> (FC_BIAS, fc_bias ,FC_OUTPUT_SIZE);

  printf("INFO: Finished allocating memory and loading weights\n");
  printf ("INFO: Start Inference\n");

#ifdef __SDSCC__
  perf_counter f_ctr;
#endif

#ifdef PROFILING
  struct timespec start, finish;
  clock_gettime(CLOCK_REALTIME, &start);
#endif

#ifdef __SDSCC__
  f_ctr.start();
#endif

  wrapper_inference(
      forget_gate_kernel_last_state_1, forget_gate_kernel_input_state_1,
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

#ifdef __SDSCC__
  f_ctr.stop();
#endif

#ifdef PROFILING
  clock_gettime(CLOCK_REALTIME, &finish);

  long seconds = finish.tv_sec - start.tv_sec;
  long ns = finish.tv_nsec - start.tv_nsec;

  if (start.tv_nsec > finish.tv_nsec) { // clock underflow
	  --seconds;
	  ns += 1000000000;
  }

  printf("seconds: %ld\n", seconds);
  printf("nanoseconds: %ld\n", ns);
  printf("total seconds: %e\n", (double)seconds + (double)ns/(double)1000000000);
#endif

#ifdef __SDSCC__
  printf("INFO:   cpu cycles %lu\n\r", f_ctr.avg_cpu_cycles());
  f_ctr.reset();
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
