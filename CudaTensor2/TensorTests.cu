#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

#include <functional>
#include <iostream>
#include <cstdio>
#include <vector>
#include <cstddef>
#include <chrono>
#include <algorithm>

#include "Tensor.cu"

using namespace std;
using namespace chrono;

std::mt19937 rng{unsigned int(std::chrono::system_clock::now().time_since_epoch().count())};

const size_t MAX_SIZE = 10000;

uniform_int<size_t> unisize(1, MAX_SIZE);
uniform_real<double> unireal(0, 1);


__host__ __device__
void Random(float& out)
{
	out = unireal(rng);
}

template <class Method, Method method, class Tensor>
void RandomInit(vector<size_t> dims, Tensor& tensor)
{
	tensor.Resize(dims);
	for (size_t i = 0; i < tensor.Volume(); i++)
		method(tensor.memory[i]);
}

template <class Method, Method method, class Tensor, class... Tensors>
void RandomInit(vector<size_t> dims, Tensor& tensor, Tensors&... tensors)
{
	RandomInit<Method, method>(dims, tensor);
	RandomInit<Method, method>(dims, tensors...);
}


void ApplyUnitTest()
{
	high_resolution_clock::time_point t0, t1;

	Tensor<> x, a, b;
	RandomInit<decltype(Random)*, Random>({MAX_SIZE, 128}, x, a, b);

	t0 = high_resolution_clock::now();
	Tensor<>::Subtract(x[{}], {128}, a[{}], {128}, b[{}], {128}, MAX_SIZE);
	t1 = high_resolution_clock::now();
	printf("Subtract CPU: %lli\n", (t1 - t0).count());

	t0 = high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		Tensor<>::Subtract(x[{}], {128}, a[{}], {128}, b[{}], {128}, MAX_SIZE);
	t1 = high_resolution_clock::now();
	printf("Subtract CPU: %lli\n", (t1 - t0).count());


	t0 = high_resolution_clock::now();
	Tensor<>::Subtract(x[{}], {128}, a[{}], {128}, b[{}], {128}, MAX_SIZE, 0);
	t1 = high_resolution_clock::now();
	printf("Subtract GPU: %lli\n", (t1 - t0).count());

	t0 = high_resolution_clock::now();
	for (size_t i = 0; i < 10; i++)
		Tensor<>::Subtract(x[{}], {128}, a[{}], {128}, b[{}], {128}, MAX_SIZE, 0);
	t1 = high_resolution_clock::now();
	printf("Subtract GPU: %lli\n", (t1 - t0).count());
}

void SortUnitTest()
{
	high_resolution_clock::time_point t0, t1;

	Tensor<> x, temp({8, 2}, 0);
	RandomInit<decltype(Random)*, Random>({8, 2}, x);

	//x.PrintToConsole("x");

	t0 = high_resolution_clock::now();
	Tensor<>::Sort(x[{}], temp[{}], x.dims[1], x.dims[0]);
	t1 = high_resolution_clock::now();
	printf("Sort GPU: %llu\n", (t1 - t0).count());

	//t0 = high_resolution_clock::now();
	//Tensor<>::Sort(x[{}], temp[{}], x.dims[1], x.dims[0], 0);
	//t1 = high_resolution_clock::now();
	//printf("Sort GPU: %llu\n", (t1 - t0).count());

	x.PrintToConsole("x");
}

void BroadcastProductTest()
{
	high_resolution_clock::time_point t0, t1;

	size_t IX = 128;
	size_t IY = 128;
	size_t Z = 3;
	size_t Wx = 8;
	size_t Wy = 8;
	size_t WX = IX / Wx;
	size_t WY = IY / Wy;
	size_t F = 64;

	Tensor<> response({WX, WY, F}, 1);
	Tensor<> weights({Wx, Wy, Z, F}, 1);
	Tensor<> windows({Wx, Wy, Z, WX, WY}, 0);

	t0 = high_resolution_clock::now();
	Tensor<>::BroadcastProduct(windows[{}],
							   dim3(WX, WY),
							   weights[{}],
							   dim3(1, 1, F),
							   weights.Volume(3),
							   response[{}],
							   dim3(WX, WY, F),
							   0);
	t1 = high_resolution_clock::now();
	printf("Broadcast Product GPU: %lli\n", (t1 - t0).count());

	t0 = high_resolution_clock::now();
	Tensor<>::BroadcastProduct(windows[{}],
							   dim3(WX, WY),
							   weights[{}],
							   dim3(1, 1, F),
							   weights.Volume(3),
							   response[{}],
							   dim3(WX, WY, F));
	t1 = high_resolution_clock::now();
	printf("Broadcast Product CPU: %lli\n", (t1 - t0).count());
}

void OuterProductTest()
{
	high_resolution_clock::time_point t0, t1;

	Tensor<> out({8, 8}, 0);
	Tensor<> v1({8}, 1);
	Tensor<> v2({8}, 1);

	t0 = high_resolution_clock::now();
	Tensor<>::OuterProduct(out[{}],
						   dim3(),
						   v1[{}],
						   dim3(),
						   v1.dims[0],
						   v2[{}],
						   dim3(),
						   v2.dims[0],
						   0);
	t1 = high_resolution_clock::now();
	printf("OuterProduct GPU: %lli\n", (t1 - t0).count());

	t0 = high_resolution_clock::now();
	Tensor<>::OuterProduct(out[{}],
						   dim3(),
						   v1[{}],
						   dim3(),
						   v1.dims[0],
						   v2[{}],
						   dim3(),
						   v2.dims[0]);
	t1 = high_resolution_clock::now();
	printf("OuterProduct CPU: %lli\n", (t1 - t0).count());
}

void InnerProductTests()
{
	high_resolution_clock::time_point t0, t1;

	Tensor<float> activation({1024}, 0);
	Tensor<float> input({4096}, 1);
	Tensor<float> weights({4096, 1024}, 1);

	t0 = high_resolution_clock::now();
	Tensor<float>::InnerProduct(activation[{}],
		                        dim3(activation.Volume()),
								input[{}],
								dim3(),
								weights[{}],
								dim3(weights.dims[1]),
								input.dims[0], 
								0);

	t1 = high_resolution_clock::now();
	printf("InnerProduct GPU: %lli\n", (t1 - t0).count());

	t0 = high_resolution_clock::now();
	Tensor<float>::InnerProduct(activation[{}],
								dim3(activation.Volume()),
								input[{}],
								dim3(),
								weights[{}],
								dim3(weights.dims[1]),
								input.dims[0],
								0);

	t1 = high_resolution_clock::now();
	printf("InnerProduct CPU: %lli\n", (t1 - t0).count());
}
