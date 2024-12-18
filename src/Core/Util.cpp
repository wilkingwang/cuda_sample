#include <iostream>
#include <type_traits>

#include "Util.h"

void GenRandomMatrix(std::vector<float>& matrix, size_t M, size_t N)
{
	srand(time(nullptr));

	float a = 5.0f;
	for (size_t i = 0; i < M; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			//matrix[i * N + j] = (float)rand() / ((float)RAND_MAX / a);
			matrix[i * N + j] = i * N + j + 1.0;
		}
	}
}

template <class T>
void CopyMatrix(const std::vector<T>& src, std::vector<T>& dst, size_t M, size_t N)
{
	for (size_t i = 0; i < M * N; i++)
	{
		dst[i] = src[i];
	}
}