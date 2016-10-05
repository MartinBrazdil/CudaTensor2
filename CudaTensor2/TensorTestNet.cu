#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
using namespace std;

#include "Tensor.cu"
#include "Network.cu"

template <class Neuron>
class BiasedPerceptron
{
public:
	std::mt19937 rng{unsigned int(std::chrono::system_clock::now().time_since_epoch().count())};

	double alpha;

	Tensor<> weights, weights_gradient, weights_gradient_rms, bias, bias_weights, bias_weights_delta, bias_weights_gradient_rms,
		activations, bias_activations, responses, gradients, gradients_delta, input_delta, output_delta;

	BiasedPerceptron(size_t dim0, size_t dim1, double alpha)
		: weights{{dim0, dim1}}, bias_weights{{1, dim1}}, alpha{alpha}
	{
		// TODO: TENSOR INITIALIZER?
		auto normal = std::normal_distribution<double>(0, 0.001);
		for (size_t i = 0; i < bias_weights.Volume(); i++)
			bias_weights.memory[i] = normal(rng);
		bias_weights.NormalizeToUnitCircle();
		for (size_t i = 0; i < weights.Volume(); i++)
			weights.memory[i] = normal(rng);
		weights.NormalizeToUnitCircle();
	}

	Tensor<>& Feedforward(Tensor<>& inputs)
	{
		size_t batch_size = inputs.dims.back();
		bias.Resize({1, batch_size}, 1);
		Activation(activations, weights, inputs);
		Activation(bias_activations, bias_weights, bias);
		activations += bias_activations;
		Responses<Neuron>(responses, activations);
		return responses;
	}

	Tensor<>& Backprop(Tensor<>& inputs, Tensor<>& output_delta)
	{
		size_t batch_size = inputs.dims.back();
		Gradients<Neuron>(gradients, responses);
		GradientDelta(gradients_delta, gradients, output_delta);
		WeightsDelta(bias_weights_delta, bias, gradients_delta);
		WeightsDelta(weights_gradient, inputs, gradients_delta);
		DeltaBack(input_delta, weights, output_delta);
		SGDUpdate(weights, weights_gradient, alpha / batch_size);
		//SGDUpdate(bias_weights, bias_weights_delta, alpha / batch_size);
		//RMSUpdate(weights, weights_gradient, weights_gradient_rms, alpha / batch_size);
		//RMSUpdate(bias_weights, bias_weights_delta, bias_weights_gradient_rms, alpha / batch_size);
		NormalizeWeights(weights);
		//NormalizeWeights(bias_weights);
		return input_delta;
	}
};

