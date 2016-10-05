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
using namespace std;
using namespace chrono;

#include "TensorTests.cu"
#include "TensorTestNet.cu"
#include "Neurons.cu"
#include "MNIST.cu"

double MSE(Tensor<> tensor)
{
	double sum = 0;
	for (size_t i = 0; i < tensor.Volume(); i++)
		sum += pow(tensor.memory[i], 2) / tensor.Volume();
	return sum;
}

template <class Network>
void LearningAlgorithms(Network& network)
{
	MNIST mnist("C:\\Gwynbleidd\\MNIST\\", 28, 28, 500, 100, true, true);
	cudaDeviceSynchronize();

	size_t batch_size = 1;

	Tensor<> mean, variance;
	Tensor<> train_inputs{{784, batch_size}}, train_targets{{10, batch_size}};
	Tensor<> test_inputs{{784, batch_size}}, test_targets{{10, batch_size}};

	while (true)
	{
		Tensor<> error, total_error;
		for (size_t i = 0; i < mnist.train_count; i++)
		{
			size_t batch_index = i % batch_size;

			train_inputs.Resize({784, batch_size}), train_targets.Resize({10, batch_size});
			Tensor<>::Copy(train_inputs[{0, batch_index}], mnist.data[i].input[{}], 784);
			Tensor<>::Copy(train_targets[{0, batch_index}], mnist.data[i].target[{}], 10);

			if (batch_index == batch_size - 1)
			{
				//Mean(mean, train_inputs);
				//Variance(variance, mean, train_inputs);
				//Standardize(variance, mean, train_inputs);

				network.Feedforward(train_inputs);
				Error(error, network.responses, train_targets);
				network.Backprop(train_inputs, error);
			}
		}

		size_t correct = 0;
		for (size_t i = 0; i < mnist.test_count; i++)
		{
			size_t batch_index = i % batch_size;

			test_inputs.Resize({784, batch_size}), test_targets.Resize({10, batch_size});
			Tensor<>::Copy(test_inputs[{0, batch_index}], mnist.data[mnist.train_count + i].input[{}], 784);
			Tensor<>::Copy(test_targets[{0, batch_index}], mnist.data[mnist.train_count + i].target[{}], 10);

			if (batch_index == batch_size - 1)
			{
				//Mean(mean, train_inputs);
				//Variance(variance, mean, train_inputs);
				//Standardize(variance, mean, train_inputs);

				network.Feedforward(test_inputs);

				Error(error, network.responses, test_targets);
				total_error += error;

				for (size_t b = 0; b < batch_size; b++)
					correct += Tensor<>::CheckClass(network.responses[{0, b}], test_targets[{0, b}], test_targets.Volume(-1));
			}
		}

		printf("total error: %f\n", MSE(total_error));
		printf("total correct: %llu\n", correct);
	}
}


class Test
{
public:
	static __device__ void device(int& a, int& b)
	{
		printf("device method\n");
	}
};

template <class Fn, Fn fn, class... Args>
__global__ void apply(Args... args)
{
	fn(args...);
}

class Test2
{
public:
	template <class Fn, Fn fn, class... Args>
	static void ComputeDevice(Args&&... args)
	{
		apply<Fn, fn><<<1, 1>>>(args...);
	}
};


int main()
{
	high_resolution_clock::time_point t0, t1;

	Test2::ComputeDevice<decltype(&Test::device), Test::device>(1, 2);

	//SortUnitTest();
	//ApplyUnitTest();

	//Tensor<> test({100});
	//Tensor<>::Apply<Identity<float>::Response>(test[{}], dim3(), test[{}], dim3(), 100);

	//Tensor<>::InnerProduct({test[{}], {}},
	//                       {test[{}], dim3(1,2,3)},
	//					   {test[{}], dim3()},
	//					   128);

	//BiasedPerceptron<Neuron<Identity<float>>>biased_perceptron(784, 10, 0.0001);
	//LearningAlgorithms(biased_perceptron);

	//BroadcastProductTest();
	//OuterProductTest();
	//InnerProductTests();

	cudaProfilerStop();
	std::cin.get();
	return 0;
}
