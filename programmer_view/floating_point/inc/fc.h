#pragma once

#include "types.h"

// fully-connected layer with fixed size of input and output vectors,
//  i.e. FC_INPUT_SIZE and FC_OUTPUT_SIZE separately
void fc(const FDATA_T* fc_input_feature_map, const FDATA_T* fc_kernel,
        const FDATA_T* fc_bias, FDATA_T* fc_output_feature_map);
