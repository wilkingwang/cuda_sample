#pragma once

unsigned int getBlockSize();
unsigned int blocksForSize(unsigned int n, unsigned int blockSize);

void reduction_global_mem(unsigned int n, unsigned int blockSize, float *d_x, float* d_y);

void reducation_shared_mem(unsigned int n, unsigned int blockSize, float* d_x, float* d_y);