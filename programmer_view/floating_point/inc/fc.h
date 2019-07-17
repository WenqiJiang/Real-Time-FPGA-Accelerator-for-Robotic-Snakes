#pragma once

#include "constants.h"
#include "types.h"

// fully-connected layer with fixed size of input and output vectors,
//  i.e. FC_INPUT_SIZE and FC_OUTPUT_SIZE separately
void fc(const FDATA_T fc_input_feature_map[LSTM_STATE_SIZE_2],
        const FDATA_T fc_kernel[FC_OUTPUT_SIZE * FC_INPUT_SIZE],
        const FDATA_T fc_bias[FC_OUTPUT_SIZE],
        FDATA_T fc_output_feature_map[FC_OUTPUT_SIZE]);
