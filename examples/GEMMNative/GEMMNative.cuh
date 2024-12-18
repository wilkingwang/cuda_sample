#pragma once
#include <cuda_runtime.h>

void sgemm(size_t M, size_t N, size_t K, float* a, float* b, float* c, float alpha = 1.0, float beta = 0);