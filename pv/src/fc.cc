#include "fc.h"

template<>
void fc(FT* fc_input_feature_map, FT* fc_kernel, FT* fc_bias, 
        FT* fc_output_feature_map) {
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
