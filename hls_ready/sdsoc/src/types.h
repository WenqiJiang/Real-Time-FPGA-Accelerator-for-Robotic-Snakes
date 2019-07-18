#pragma once

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_math.h"

#define IDATA_T short
#define LDATA_T int

#define FXD_W_LENGTH 16
#define FXD_I_LENGTH 7
#define FDATA_T ap_fixed<FXD_W_LENGTH,FXD_I_LENGTH>

#define TOFLOAT(x) x.to_float()
#define TOINT(x) int(x)

#ifdef __SDSCC__
#include "sds_lib.h"

class perf_counter {
  public:
    long unsigned tot, cnt, calls;
    perf_counter() : tot(0), cnt(0), calls(0) {};
    inline void reset() { tot = cnt = calls = 0; }
    inline void start() { cnt = sds_clock_counter(); calls++; };
    inline void stop() { tot += (sds_clock_counter() - cnt); };
    inline long unsigned avg_cpu_cycles() { return (tot / calls); };
};
#endif

#define MALLOC sds_alloc
#define MFREE sds_free
