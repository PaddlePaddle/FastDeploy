
// The code is based on:
// https://github.com/gatagat/lap/blob/master/lap/lapjv.h
// Ths copyright of gatagat/lap is as follows:
// MIT License

#ifndef DEPLOY_PPTRACKING_CPP_INCLUDE_LAPJV_H_
#define DEPLOY_PPTRACKING_CPP_INCLUDE_LAPJV_H_
#define LARGE 1000000

#if !defined TRUE
#define TRUE 1
#endif
#if !defined FALSE
#define FALSE 0
#endif

#define NEW(x, t, n)                                               \
  if ((x = reinterpret_cast<t *>(malloc(sizeof(t) * (n)))) == 0) { \
    return -1;                                                     \
  }
#define FREE(x) \
  if (x != 0) { \
    free(x);    \
    x = 0;      \
  }
#define SWAP_INDICES(a, b) \
  {                        \
    int_t _temp_index = a; \
    a = b;                 \
    b = _temp_index;       \
  }
#include <opencv2/opencv.hpp>

typedef signed int int_t;
typedef unsigned int uint_t;
typedef double cost_t;
typedef char boolean;
typedef enum fp_t { FP_1 = 1, FP_2 = 2, FP_DYNAMIC = 3 } fp_t;

int lapjv_internal(const cv::Mat &cost,
                   const bool extend_cost,
                   const float cost_limit,
                   int *x,
                   int *y);


#endif  // DEPLOY_PPTRACKING_CPP_INCLUDE_LAPJV_H_
