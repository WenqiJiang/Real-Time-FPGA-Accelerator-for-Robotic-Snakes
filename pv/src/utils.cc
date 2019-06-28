#include "utils.h"                                                              
                                                                                
#include "types.h"                                                              
                                                                                
template<>                                                                                          
void zero_init(FDATA_T* input_array, LDATA_T array_length)                      
{                                                                               
    for(LDATA_T idx = 0; idx < array_length; idx++)                             
        input_array[idx] = 0;                                                   
} 
