#include "fc.h"

#include "constants.h"
#include "types.h"

void fc(const FDATA_T fc_input_feature_map[LSTM_STATE_SIZE_2],
        const FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        const FDATA_T fc_bias[FC_OUTPUT_SIZE],
        FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE]) {
  // input:
  // fc_input_feature_map: a vector with a size of FC_INPUT_SIZE
  // fc_kernel: FC_OUTPUT_SIZE x FC_INPUT_SIZE, notice that this kernel is
  //  TRANSPOSED.
  // fc_bias: FC_OUTPUT_SIZE
  // output:
  // fc_output_feature_map: a vector with a size of FC_OUTPUT_SIZE

  for (IDATA_T result_idx = 0; result_idx < FC_OUTPUT_SIZE; result_idx++) {

    // initialization
    fc_output_feature_map[result_idx] = 0;

    // matrix multiplication
    for (IDATA_T sum_idx = 0; sum_idx < FC_INPUT_SIZE; sum_idx++) {

      IDATA_T fc_kernel_idx = result_idx * FC_INPUT_SIZE + sum_idx;

      fc_output_feature_map[result_idx] += fc_kernel[fc_kernel_idx] *
                                           fc_input_feature_map[sum_idx];
    }

    // add bias
    fc_output_feature_map[result_idx] += fc_bias[result_idx];
  }
}
