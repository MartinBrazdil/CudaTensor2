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
#include "Neurons.cu"

static void Activation(Tensor<>& activations, Tensor<>& weights, Tensor<>& inputs)
{
	size_t batch_size = inputs.dims.back();
	activations.Resize({weights.dims[1], batch_size});

	activations.FillValue(0, 0);
	Tensor<>::InnerProduct(activations[{}],
						   dim3(weights.dims[1], batch_size),
						   weights[{}],
						   dim3(weights.dims[1]),
						   inputs[{}],
						   dim3(1, batch_size),
						   weights.dims[0],
						   0);
}

template <class Neuron>
static void Responses(Tensor<>& responses, Tensor<>& activations)
{
	responses.Resize(activations.dims);
	Neuron::Response(responses[{}], dim3(), activations[{}], dim3(), activations.Volume(), 0);
}

template <class Neuron>
static void Gradients(Tensor<>& gradients, Tensor<>& responses)
{
	gradients.Resize(responses.dims);
	Neuron::Gradient(gradients[{}], dim3(), responses[{}], dim3(), responses.Volume(), 0);
}

static void Error(Tensor<>& errors, Tensor<>& output, Tensor<>& targets)
{
	errors.Resize(targets.dims);
	Tensor<>::Subtract(errors[{}],
					   dim3(),
					   output[{}],
					   dim3(),
					   targets[{}],
				 	   dim3(),
					   errors.Volume());
}

static void GradientDelta(Tensor<>& gradients_delta, Tensor<>& gradients, Tensor<>& output_delta)
{
	gradients_delta.Resize(gradients.dims);
	Tensor<>::Multiply(gradients_delta[{}],
					   gradients[{}],
					   output_delta[{}],
					   gradients_delta.Volume());
}

static void WeightsDelta(Tensor<>& weights_gradient, Tensor<>& inputs, Tensor<>& gradients_delta)
{
	size_t batch_size = inputs.dims.back();
	weights_gradient.Resize({inputs.Volume(-1), gradients_delta.Volume(-1)});

	Tensor<>::OuterProduct(weights_gradient[{}],
						   dim3(weights_gradient.dims[1]),
						   gradients_delta[{}],
						   dim3(1, batch_size),
						   weights_gradient.dims[1],
						   inputs[{}],
						   dim3(1, batch_size),
						   weights_gradient.dims[0],
						   0);
}

static void SGDUpdate(Tensor<>& weights, Tensor<>& weights_gradient, double alpha)
{
	for (size_t i = 0; i < weights.Volume(); i++)
	{
		weights.memory[i] -= alpha * weights_gradient.memory[i];
		weights_gradient.memory[i] = 0;
	}
}

static void RMSUpdate(Tensor<>& weights, Tensor<>& weights_gradient, Tensor<>& weights_gradient_rms, double alpha)
{
	const double gamma = 0.9;
	weights_gradient_rms.Resize(weights_gradient.dims, 0.00001);

	for (size_t i = 0; i < weights.Volume(); i++)
	{
		weights_gradient_rms.memory[i] = gamma * weights_gradient_rms.memory[i] + (1 - gamma) * std::pow(weights_gradient.memory[i], 2);
		double learning_rate = alpha / std::sqrt(weights_gradient_rms.memory[i]);
		weights.memory[i] -= learning_rate * weights_gradient.memory[i];
		weights_gradient.memory[i] = 0;
	}
}

static void ADUpdate(Tensor<>& weights, Tensor<>& weights_gradient, Tensor<>& weights_gradient_rms, Tensor<>& weights_delta_rms)
{
	const double gamma = 0.9;
	weights_gradient_rms.Resize(weights_gradient.dims, 0.00001);
	weights_delta_rms.Resize(weights_gradient.dims, 0.00001);

	for (size_t i = 0; i < weights.Volume(); i++)
	{
		weights_gradient_rms.memory[i] = gamma * weights_gradient_rms.memory[i] + (1 - gamma) * std::pow(weights_gradient.memory[i], 2);
		double learning_rate = std::sqrt(weights_delta_rms.memory[i]) / std::sqrt(weights_gradient_rms.memory[i]);
		double weight_delta = -learning_rate * weights_gradient.memory[i];
		//printf("%f + %f\n", gamma * weights_delta_rms.memory[i], (1 - gamma) *  std::pow(weight_delta, 2));
		weights_delta_rms.memory[i] = gamma * weights_delta_rms.memory[i] + (1 - gamma) *  std::pow(weight_delta, 2);
		weights.memory[i] += weight_delta;
		weights_gradient.memory[i] = 0;
	}
}

static void NormalizeWeights(Tensor<>& weights)
{
	for (size_t o = 0; o < weights.dims[1]; o++)
		Tensor<>::NormalizeToUnitCircle(weights[{0, o}], weights.Volume(1));
}

static void DeltaBack(Tensor<>& input_delta, Tensor<>& weights, Tensor<>& output_delta)
{
	size_t batch_size = output_delta.dims.back();
	input_delta.Resize({weights.dims[0], batch_size});

	input_delta.FillValue(0);
	Tensor<>::InnerProduct(input_delta[{}],
						   dim3(1, batch_size),
						   weights[{}],
						   dim3(weights.dims[1]),
						   output_delta[{}],
						   dim3(1, batch_size),
						   weights.dims[1],
						   0);
}

static void Mean(Tensor<>& mean, Tensor<>& batch)
{
	mean.Resize({batch.Volume(-1)});
	for (size_t i = 0; i < batch.dims.back(); i++)
		Tensor<>::Add(mean[{}], mean[{}], batch[{0, i}], mean.Volume());
	mean /= batch.dims.back();
}

static void Variance(Tensor<>& variance, Tensor<>& mean, Tensor<>& batch)
{
	variance.Resize({batch.Volume(-1)});
	for (size_t i = 0; i < batch.dims.back(); i++)
		Tensor<>::Deviation(variance[{}], batch[{0, i}], mean[{}], variance.Volume());
	variance /= (batch.dims.back() - 1);
}

static void Standardize(Tensor<>& variance, Tensor<>& mean, Tensor<>& batch)
{
	Tensor<>::Sqrt(variance[{}], variance[{}], variance.Volume());
	for (size_t i = 0; i < batch.dims.back(); i++)
	{
		//Tensor<>::Subtract(batch[{0, i}], dim3(), batch[{0, i}], dim3(), mean[{}], dim3(), mean.Volume());
		Tensor<>::Divide(batch[{0, i}], batch[{0, i}], variance[{}], variance.Volume());
	}
}
