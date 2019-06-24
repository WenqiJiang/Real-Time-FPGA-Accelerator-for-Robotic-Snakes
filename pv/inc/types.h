#pragma once

#define IDATA_T int

#ifdef __SDSOC__

#include "ap_int.h" 
#include "ap_fixed.h"
#include "hls_math.h"

#define FXD_W_LENGTH 16
#define FXD_I_LENGTH 7
#define FDATA_T ap_fixed<FXD_W_LENGTH,FXD_I_LENGTH>
#define TOFLOAT(x) x.to_float()

#else
#define FDATA_T float
#endif
