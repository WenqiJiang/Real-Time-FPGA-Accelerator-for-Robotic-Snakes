#pragma once

// fully-connected layer with fixed size of input and output vectors, 
//  i.e. FC_INPUT_SIZE and FC_OUTPUT_SIZE separately
void fc(FT* fc_input_feature_map, FT* fc_kernel, FT* fc_bias, 
        FT* fc_output_feature_map);
