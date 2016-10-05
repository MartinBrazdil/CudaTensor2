#pragma once

#include "Tensor.cu"

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void NeuronKernel(T* out_ptr, dim3 out_dims, T* vec_ptr, dim3 vec_dims, size_t vec_size)
{
	float* out = &out_ptr[grid_vector(out_dims, vec_size)];
	float* vec = &vec_ptr[grid_vector(vec_dims, vec_size)];

	for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
		device_method(out[i], vec[i]);
}

template <class T, class HostMethod, HostMethod host_method>
void NeuronHost(T* out_ptr, dim3 out_dims, T* vec_ptr, dim3 vec_dims, size_t vec_size)
{
	dim3 dims = dims_max(out_dims, vec_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, vec_size, x, y, z)];
				T* vec = &vec_ptr[grid_vector(vec_dims, vec_size, x, y, z)];

				for (size_t i = threadIdx.x; i < vec_size; i++)
					host_method(out[i], vec[i]);
			}
}

template <class NeuronType>
class Neuron : public NeuronType
{
public:
	static void Response(Tensor<>::iterator res, dim3 res_dims, Tensor<>::iterator act, dim3 act_dims, size_t vec_size)
	{
		NeuronHost<NeuronType::type, decltype(NeuronType::Response)*, NeuronType::Response>(&*res, res_dims, &*act, act_dims, vec_size);
	}

	static void Response(Tensor<>::iterator res, dim3 res_dims, Tensor<>::iterator act, dim3 act_dims, size_t vec_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(res_dims, act_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		NeuronKernel<NeuronType::type, decltype(NeuronType::Response)*, NeuronType::Response><<<dims, threads>>>(&*res, res_dims, &*act, act_dims, vec_size);
		cudaDeviceSynchronize();
	}

	static void Gradient(Tensor<>::iterator res, dim3 res_dims, Tensor<>::iterator act, dim3 act_dims, size_t vec_size)
	{
		NeuronKernel<NeuronType::type, decltype(NeuronType::Gradient)*, NeuronType::Gradient>(&*res, res_dims, &*act, act_dims, vec_size);
	}

	static void Gradient(Tensor<>::iterator res, dim3 res_dims, Tensor<>::iterator act, dim3 act_dims, size_t vec_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(res_dims, act_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		NeuronKernel<NeuronType::type, decltype(NeuronType::Gradient)*, NeuronType::Gradient><<<dims, threads>>>(&*res, res_dims, &*act, act_dims, vec_size);
		cudaDeviceSynchronize();
	}
};

template <class T = float>
struct Identity
{
	using type = T;

	static __device__ __host__
	void Response(T& res, T& act)
	{
		res = act;
	}

	static __device__ __host__
	void Gradient(T& grad, T& res)
	{
		grad = 1;
	}
};

template <class T = float>
struct Logistic
{
	using type = T;

	static __host__ __device__
	T Response(T& res, T& act)
	{
		res = 1.0 / (1.0 + exp(-(act / 1.0)));
	}

	static __host__ __device__
	T Gradient(T& grad, T& res)
	{
		grad =  res * (1 - res);
	}
};

template <class T = float>
struct LecunTanh
{
	using type = T;

	static __host__ __device__
	T Response(T& res, T& act)
	{
		res = 1.7159 * tanh((2.0 * act) / 3.0);
	}

	static __host__ __device__
	T Gradient(T& grad, T& res)
	{
		res = (2.0 * res) / 3.0;
		res = 2 / (exp(res) + exp(-res));
		grad = 1.14393 * pow(res, 2);
	}
};

template <class T = float>
struct ReLU
{
	using type = T;

	static __host__ __device__
	T Response(T& res, T& act)
	{
		res = std::max(0.0, act);
	}

	static __host__ __device__
	T Gradient(T& grad, T& res)
	{
		grad = res > 0 ? res : 0.01 * res;
	}
};

template <class T = float>
struct Softplus
{
	using type = T;

	static __host__ __device__
	T Response(T& res, T& act)
	{
		res = log(1.0 + exp(act));
	}

	static __host__ __device__
	T Gradient(T& grad, T& res)
	{
		grad = 1.0 / (1.0 + exp(-(res / 1.0)));
	}
};
