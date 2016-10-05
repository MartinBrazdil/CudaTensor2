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
#include <cstdlib>
using namespace std;

template <class T>
class cuda_allocator
{
public:
	using value_type = T;

	cuda_allocator() {};

	template <class U>
	cuda_allocator(const cuda_allocator<U>& other)
		: cuda_allocator<T>()
	{}

	T* allocate(size_t n)
	{
		T* memory;
		cudaMallocManaged(&memory, n * sizeof(T));
		cudaDeviceSynchronize();
		return memory;
	}

	void deallocate(T* p, size_t n)
	{
		cudaFree(p);
		cudaDeviceSynchronize();
	}
};

static dim3 dims_max(dim3 d1, dim3 d2)
{
	return dim3(max(d1.x, d2.x), max(d1.y, d2.y), max(d1.z, d2.z));
}

template <class T, class... Ts>
static dim3 dims_max(T d1, T d2, Ts... ds)
{
	return dims_max(dims_max(d1, d2), ds...);
}

inline size_t grid_vector(dim3 dims, size_t vec_size, size_t x, size_t y, size_t z)
{
	return ((z % dims.z) * dims.y * dims.x +
			(y % dims.y) * dims.x +
			(x % dims.x)) * vec_size;
};

__device__ __forceinline__ size_t grid_vector(dim3 dims, size_t vec_size)
{
	return ((blockIdx.z % dims.z) * dims.y * dims.x +
			(blockIdx.y % dims.y) * dims.x +
			(blockIdx.x % dims.x)) * vec_size;
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void InnerKernel(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, T* vec2_ptr, dim3 vec2_dims, size_t vec_size)
{
	T& out = out_ptr[grid_vector(out_dims, 1)];
	T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec_size)];
	T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec_size)];

	extern __shared__ T partials[];
	size_t tid = threadIdx.x;

	partials[tid] = 0;
	for (size_t i = tid; i < vec_size; i += blockDim.x)
		device_method(partials[tid], vec1[i], vec2[i]);
	__syncthreads();

	for (size_t i = 1024; 1 < i && tid < i / 2; i /= 2)
		if (blockDim.x >= i)
		{
			partials[tid] += partials[tid + i / 2];
			device_method(partials[tid], partials[i], 0);
			__syncthreads();
		}

	if (tid == 0)
		out = partials[0];
}

template <class T, class HostMethod, HostMethod host_method>
void InnerHost(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, T* vec2_ptr, dim3 vec2_dims, size_t vec_size)
{
	dim3 dims = dims_max(out_dims, vec_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T& out = out_ptr[grid_vector(out_dims, 1, x, y, z)];
				T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec_size, x, y, z)];
				T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec_size, x, y, z)];

				for (size_t i = 0; i < vec_size; i++)
					host_method(out, vec1[i], vec2[i]);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void OuterKernel(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, size_t vec1_size, T* vec2_ptr, dim3 vec2_dims, size_t vec2_size)
{
	T* out = &out_ptr[grid_vector(out_dims, vec2_size)];
	T& vec1 = vec1_ptr[grid_vector(vec1_dims, 1)];
	T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec2_size)];

	for (size_t i = threadIdx.x; i < vec2_size; i += blockDim.x)
		device_method(out[i], vec1, vec2[i]);
}

template <class T, class HostMethod, HostMethod host_method>
void OuterHost(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, size_t vec1_size, T* vec2_ptr, dim3 vec2_dims, size_t vec2_size)
{
	dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, vec2_size, x, y, z)];
				T& vec1 = vec1_ptr[grid_vector(vec1_dims, 1, x, y, z)];
				T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec2_size, x, y, z)];

				for (size_t i = 0; i < vec2_size; i++)
					host_method(out[i], vec1, vec2[i]);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void BroadcastKernel(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, size_t vec1_size, T* vec2_ptr, dim3 vec2_dims)
{
	T* out = &out_ptr[grid_vector(out_dims, vec1_size)];
	T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec1_size)];
	T& vec2 = vec2_ptr[grid_vector(vec2_dims, 1)];

	for (size_t i = threadIdx.x; i < vec1_size; i += blockDim.x)
		device_method(out[i], vec1[i], vec2);
}

template <class T, class HostMethod, HostMethod host_method>
void BroadcastHost(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, size_t vec1_size, T* vec2_ptr, dim3 vec2_dims)
{
	dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, vec1_size, x, y, z)];
				T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec1_size, x, y, z)];
				T& vec2 = vec2_ptr[grid_vector(vec2_dims, 1, x, y, z)];

				for (size_t i = 0; i < vec1_size; i++)
					host_method(out[i], vec1[i], vec2);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void ApplyKernel(T* out_ptr, dim3 out_dims, size_t out_size)
{
	T* out = &out_ptr[grid_vector(out_dims, out_size)];

	for (size_t i = threadIdx.x; i < out_size; i += blockDim.x)
		device_method(out[i]);
}

template <class T, class HostMethod, HostMethod host_method>
void ApplyHost(T* out_ptr, dim3 out_dims, size_t out_size)
{
	dim3 dims = out_dims;

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, out_size, x, y, z)];

				for (size_t i = 0; i < out_size; i++)
					host_method(out[i]);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void ApplyKernel(T* out_ptr, dim3 out_dims, size_t out_size, T value)
{
	T* out = &out_ptr[grid_vector(out_dims, out_size)];

	for (size_t i = threadIdx.x; i < out_size; i += blockDim.x)
		device_method(out[i], value);
}

template <class T, class HostMethod, HostMethod host_method>
void ApplyHost(T* out_ptr, dim3 out_dims, size_t out_size, T value)
{
	dim3 dims = out_dims;

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, out_size, x, y, z)];

				for (size_t i = 0; i < out_size; i++)
					host_method(out[i], value);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void ApplyKernel(T* out_ptr, dim3 out_dims, T* vec_ptr, dim3 vec_dims, size_t vec_size)
{
	T* out = &out_ptr[grid_vector(out_dims, vec_size)];
	T* vec = &vec_ptr[grid_vector(vec_dims, vec_size)];

	for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
		device_method(out[i], vec[i]);
}

template <class T, class HostMethod, HostMethod host_method>
void ApplyHost(T* out_ptr, dim3 out_dims, T* vec_ptr, dim3 vec_dims, size_t vec_size)
{
	dim3 dims = dims_max(out_dims, vec_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, vec_size, x, y, z)];
				T* vec = &vec_ptr[grid_vector(vec_dims, vec_size, x, y, z)];

				for (size_t i = 0; i < out_size; i++)
					host_method(out[i], vec[i]);
			}
}

template <class T, class DeviceMethod, DeviceMethod device_method>
__global__ void ApplyKernel(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, T* vec2_ptr, dim3 vec2_dims, size_t vec_size)
{
	T* out = &out_ptr[grid_vector(out_dims, vec_size)];
	T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec_size)];
	T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec_size)];

	for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
		device_method(out[i], vec1[i], vec2[i]);
}

template <class T, class HostMethod, HostMethod host_method>
void ApplyHost(T* out_ptr, dim3 out_dims, T* vec1_ptr, dim3 vec1_dims, T* vec2_ptr, dim3 vec2_dims, size_t vec_size)
{
	dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);

	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* out = &out_ptr[grid_vector(out_dims, vec_size, x, y, z)];
				T* vec1 = &vec1_ptr[grid_vector(vec1_dims, vec_size, x, y, z)];
				T* vec2 = &vec2_ptr[grid_vector(vec2_dims, vec_size, x, y, z)];

				for (size_t i = 0; i < vec_size; i++)
					host_method(out[i], vec1[i], vec2[i]);
			}
}

template <class T>
__device__ void Merge(T* lvect, size_t lsize, T* rvect, size_t rsize, T* temp)
{
	for (size_t i = 0, li = 0, ri = 0; i < (lsize + rsize); i++)
		if (ri >= rsize || lvect[li] < rvect[ri] && li < lsize)
			temp[i] = lvect[li++];
		else
			temp[i] = rvect[ri++];
}

template <class T>
__device__ void Swap(T* input, T* temp, size_t size)
{
	for (size_t i = threadIdx.x; i < size; i += blockDim.x)
	{
		T dummy = input[i];
		input[i] = temp[i];
		temp[i] = dummy;
	}
}

template <class T>
__global__ void SortKernel(T* in_ptr, T* tmp_ptr, dim3 dims, int size)
{
	T* in = &in_ptr[grid_vector(dims, size)];
	T* tmp = &tmp_ptr[grid_vector(dims, size)];

	for (int d = 1; d < size; d *= 2)
	{
		for (int i = (2 * d)*threadIdx.x; i < size; i += (2 * d)*blockDim.x)
			Merge(&in[i], max(0, min(size - i, d)), &in[i + d], max(0, min(size - (i + d), d)), &tmp[i]);
		__syncthreads();
		Swap(in, tmp, size);
		__syncthreads();
	}
}

template <class T>
void SortHost(T* in_ptr, T* tmp_ptr, dim3 dims, int size)
{
	for (size_t z = 0; z < dims.z; z++)
		for (size_t y = 0; y < dims.y; y++)
			for (size_t x = 0; x < dims.x; x++)
			{
				T* in = &in_ptr[grid_vector(dims, size, x, y, z)];
				T* tmp = &tmp_ptr[grid_vector(dims, size, x, y, z)];

				sort(in, in + size);
			}
}

template <class T = float>
class Tensor
{
public:
	template <class U>
	using cuda_vector = std::vector<U, cuda_allocator<U>>;
	using iterator = typename cuda_vector<T>::iterator;

	vector<size_t> dims;
	cuda_vector<T> memory;

	Tensor()
		:dims{1}, memory(1)
	{}

	Tensor(vector<size_t> dims, const T& init = {})
		: dims{dims}
	{
		memory.resize(Volume(), init);
	}

	static void Sort(iterator in_it, iterator tmp_it, dim3 dims, size_t size, cudaStream_t stream)
	{
		size_t threads = 64 < size / 2 ? 64 : size / 2;
		SortKernel<T><<<dims, threads, 0, stream>>>(&*in_it, &*tmp_it, dims, size);
		cudaStreamSynchronize(stream);
	}

	static void Sort(iterator in_it, iterator tmp_it, dim3 dims, size_t size)
	{
		SortHost(&*in_it, &*tmp_it, dims, size);
	}

	static void PrintToConsole(iterator vec_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
			printf("%.02f, ", *vec_it++);
		printf("\n");
	}

	void PrintToConsole(string name = "")
	{
		printf("Tensor %s: ", name.c_str());
		for (size_t i = 0; i < Volume(); i++)
			printf("%.02f, ", memory[i]);
		printf("\n");
	}

	static void PrintToFile(std::string path, iterator vec_it, size_t size)
	{
		std::ofstream file(path);
		for (size_t i = 0; i < size; i++)
			file << (*vec_it++) << " ";
		file << std::endl;
		file.close();
	}

	void PrintToFile(std::string path)
	{
		std::ofstream file(path);
		for (size_t i = 0; i < Volume(); i++)
			file << memory[i] << " ";
		file << std::endl;
		file.close();
	}

	static void Copy(iterator v_out, iterator vec_it, size_t size)
	{
		std::copy(vec_it, vec_it + size, v_out);
	}

	__host__ __device__
	static void Assign(T& out, T& val)
	{
		out = val;
	}



	//template <class Method, Method method, class Tensor>
	//void RandomInit(vector<size_t> dims, Tensor& tensor)
	//{
	//	//tensor.Resize(dims);
	//	//for (size_t i = 0; i < tensor.Volume(); i++)
	//	//	tensor.memory[i] = unireal(rng);

	//}

	//template <class Method, Method method, class Tensor, class... Tensors>
	//void FillValue(vector<size_t> dims, Tensor& tensor, Tensors&... tensors)
	//{
	//	FillValue<Method, method>(dims, tensor);
	//	FillValue<Method, method>(dims, tensors...);
	//}


	void FillValue(T value)
	{
		ApplyHost<T, decltype(Assign)*, Assign>(&memory[0], dim3(), Volume(), value);
	}

	void FillValue(T value, cudaStream_t stream)
	{
		size_t threads = 1024 < Volume() ? 1024 : Volume();
		ApplyKernel<T, decltype(Assign)*, Assign><<<1, threads, 0, stream>>>(&memory[0], dim3(), Volume(), value);
		cudaStreamSynchronize(stream);
	}


	void FillLambda(void(*host_method)(T&))
	{
		ApplyHost<T, decltype(host_method)*, host_method>(&memory[0], dim3(), Volume());
	}

	void FillLambda(void(*device_method)(T&), cudaStream_t stream)
	{
		size_t threads = 1024 < Volume() ? 1024 : Volume();
		ApplyKernel<T, decltype(device_method)*, device_method><<<1, threads, 0, stream>>>(&memory[0], dim3(), Volume());
		cudaStreamSynchronize(stream);
	}


	size_t Volume(int N) const
	{
		N = N % dims.size();
		N = 0 < N ? N : N + dims.size();
		size_t volume = 1;
		for (size_t i = 0; i < N; i++)
			volume *= dims[i];
		return volume;
	}

	size_t Volume() const
	{
		return Volume(dims.size());
	}

	iterator operator[](vector<size_t> idxs)
	{
		if (idxs.empty())
			return memory.begin();

		size_t offset = idxs[0];
		size_t dim_size = dims[0];
		for (size_t i = 1; i < idxs.size(); i++)
		{
			offset += idxs[i] * dim_size;
			dim_size *= dims[i];
		}
		return memory.begin() + offset;
	}

	bool Resize(vector<size_t> dims, T value = T{})
	{
		if (!dims.empty() && this->dims != dims)
		{
			this->dims = dims;
			memory.resize(Volume(), value);
			cudaDeviceSynchronize();
			return true;
		}
		return false;
	}

	void Reshape(vector<size_t> dims)
	{
		size_t volume = Volume();
		this->dims = dims;
		if (volume != Volume())
			throw "Cannot reshape to different volume";
	}

	void Transpose(Tensor<> out, size_t n)
	{
		vector<size_t> out_dims(dims.size());
		std::copy(dims.begin(), dims.begin() + n, out_dims.begin());
		std::copy(dims.begin() + n, dims.end(), out_dims.begin() + n);

	}

	static void Add(iterator out_it, iterator vec1_it, iterator vec2_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& vec1 = (*vec1_it++);
			T& vec2 = (*vec2_it++);
			T& out = (*out_it++);
			out = vec1 + vec2;
		}
	};

	static void Add(iterator out_it, iterator vec1_it, T value, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& vec1 = (*vec1_it++);
			T& out = (*out_it++);
			out = vec1 + value;
		}
	};

	Tensor<T> operator+(const Tensor<>& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] + rhs.memory[i];
		return out;
	}

	Tensor<T>& operator+=(const Tensor<>& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] += rhs.memory[i];
		return *this;
	}

	Tensor<T> operator+(const T& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] + rhs;
		return out;
	}

	Tensor<T>& operator+=(const T& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] += rhs;
		return *this;
	}

	static __host__ __device__
	void Difference(T& out, const T& vec1, const T& vec2)
	{
		out = vec1 - vec2;
	}

	static void Subtract(iterator out, dim3 out_dims, iterator vec, dim3 vec_dims, size_t vec_size)
	{
		ApplyHost<T, decltype(Difference)*, Difference>(&*out, out_dims, &*out, out_dims, &*vec, vec_dims, vec_size);
	}

	static void Subtract(iterator out, dim3 out_dims, iterator vec, dim3 vec_dims, size_t vec_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(out_dims, vec_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		ApplyKernel<T, decltype(Difference)*, Difference><<<dims, threads, 0, stream>>>(&*out, out_dims, &*out, out_dims, &*vec, vec_dims, vec_size);
		cudaDeviceSynchronize();
	}

	static void Subtract(iterator out, dim3 out_dims, iterator vec1, dim3 vec1_dims, iterator vec2, dim3 vec2_dims, size_t vec_size)
	{
		ApplyHost<T, decltype(Difference)*, Difference>(&*out, out_dims, &*vec1, vec1_dims, &*vec2, vec2_dims, vec_size);
	}

	static void Subtract(iterator out, dim3 out_dims, iterator vec1, dim3 vec1_dims, iterator vec2, dim3 vec2_dims, size_t vec_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		ApplyKernel<T, decltype(Difference)*, Difference><<<dims, threads, 0, stream>>>(&*out, out_dims, &*vec1, vec1_dims, &*vec2, vec2_dims, vec_size);
		cudaDeviceSynchronize();
	}

	Tensor<T> operator-(const Tensor<>& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] - rhs.memory[i];
		return out;
	}

	Tensor<T>& operator-=(const Tensor<>& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] -= rhs.memory[i];
		return *this;
	}

	Tensor<T> operator-(const T& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] - rhs;
		return out;
	}

	Tensor<T>& operator-=(const T& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] -= rhs;
		return *this;
	}

	static void Multiply(iterator out_it, iterator vec1_it, iterator vec2_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& vec1 = (*vec1_it++);
			T& vec2 = (*vec2_it++);
			T& out = (*out_it++);
			out = vec1 * vec2;
		}
	};

	static void Multiply(iterator out_it, iterator vec_it, T value, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& v = (*vec_it++);
			T& out = (*out_it++);
			out = v * value;
		}
	};

	Tensor<T> operator*(const Tensor<>& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] * rhs.memory[i];
		return out;
	}

	Tensor<T>& operator*=(const Tensor<>& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] *= rhs.memory[i];
		return *this;
	}

	Tensor<T> operator*(const T& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] * rhs;
		return out;
	}

	Tensor<T>& operator*=(const T& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] *= rhs;
		return *this;
	}

	static void Divide(iterator out_it, iterator vec1_it, iterator vec2_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& vec1 = (*vec1_it++);
			T& vec2 = (*vec2_it++);
			T& out = (*out_it++);
			out = vec1 / vec2;
		}
	};

	static void Divide(iterator out_it, iterator vec_it, T value, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& v = (*vec_it++);
			T& out = (*out_it++);
			out = v / value;
		}
	};

	Tensor<T> operator/(const Tensor<>& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] / rhs.memory[i];
		return out;
	}

	Tensor<T>& operator/=(const Tensor<>& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] /= rhs.memory[i];
		return *this;
	}

	Tensor<T> operator/(const T& rhs)
	{
		Tensor<> out(dims);
		for (size_t i = 0; i < Volume(); i++)
			out.memory[i] = memory[i] / rhs;
		return out;
	}

	Tensor<T>& operator/=(const T& rhs)
	{
		for (size_t i = 0; i < Volume(); i++)
			memory[i] /= rhs;
		return *this;
	}

	static bool CheckClass(iterator output, iterator binary, size_t size)
	{
		size_t max_idx = 0;
		T max = std::numeric_limits<T>::min();

		for (size_t i = 0; i < size; i++)
		{
			T& out = (*output++);

			if (max < out)
			{
				max = out;
				max_idx = i;
			}
		}
		return *(binary + max_idx) == 1.0;
	}

	void NormalizeToInterval(T min = 0, T max = 1)
	{
		T min_val = std::numeric_limits<T>::max();
		T max_val = std::numeric_limits<T>::min();

		for (size_t i = 0; i < Volume(); i++)
		{
			min_val = std::min(min_val, memory[i]);
			max_val = std::max(max_val, memory[i]);
		}

		for (size_t i = 0; i < Volume(); i++)
			memory[i] = (max - min) * ((memory[i] - min_val) / (max_val - min_val)) + min;
	}

	static void NormalizeToInterval(iterator out_it, iterator in_it, size_t size, T min = 0, T max = 1)
	{
		T min_val = std::numeric_limits<T>::max();
		T max_val = std::numeric_limits<T>::min();

		iterator in_begin = in_it;
		for (size_t i = 0; i < size; i++)
		{
			T& in = (*in_it++);
			min_val = std::min(min_val, in);
			max_val = std::max(max_val, in);
		}

		in_it = in_begin;
		for (size_t i = 0; i < size; i++)
		{
			T& in = (*in_it++);
			T& out = (*out_it++);
			out = (max - min) * ((in - min_val) / (max_val - min_val)) + min;
		}
	}

	static void Deviation(iterator out_it, iterator vec1_it, iterator vec2_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& out = (*out_it++);
			T& vec1 = (*vec1_it++);
			T& vec2 = (*vec2_it++);

			out += std::pow(vec1 - vec2, 2);
		}
	}

	static void Sqrt(iterator out_it, iterator vec_it, size_t size)
	{
		for (size_t i = 0; i < size; i++)
		{
			T& out = (*out_it++);
			T& v = (*vec_it++);

			out = std::sqrt(v);
		}
	}

	T L2Norm()
	{
		T norm = 0;
		for (size_t i = 0; i < memory.size(); i++)
			norm += std::pow(memory[i], 2);
		return std::sqrt(norm);
	}

	static T L2Norm(iterator vec_it, size_t size)
	{
		T norm = 0;
		for (size_t i = 0; i < size; i++)
		{
			T& v = (*vec_it++);
			norm += std::pow(v, 2);
		}
		return std::sqrt(norm);
	}

	void NormalizeToUnitCircle()
	{
		T norm = L2Norm();
		for (size_t i = 0; i < memory.size(); i++)
			memory[i] /= norm;
	}

	static void NormalizeToUnitCircle(iterator vec_it, size_t size)
	{
		T norm = L2Norm(vec_it, size);
		for (size_t i = 0; i < size; i++)
		{
			T& v = (*vec_it++);
			v /= norm;
		}
	}

	__host__ __device__
	static void Product(T& out, const T vec1, const T vec2)
	{
		out += vec1 * vec2;
	}

	static void InnerProduct(iterator out_it, dim3 out_dims, iterator vec1_it, dim3 vec1_dims, iterator vec2_it, dim3 vec2_dims, size_t vec_size)
	{
		InnerKernel<T, decltype(Product)*, Product>(&*out_it, out_dims, &*vec1_it, vec1_dims, &*vec2_it, vec2_dims, vec_size);
	}

	static void InnerProduct(iterator out_it, dim3 out_dims, iterator vec1_it, dim3 vec1_dims, iterator vec2_it, dim3 vec2_dims, size_t vec_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		InnerKernel<T, decltype(Product)*, Product><<<dims, threads, threads * sizeof(T), stream>>>(&*out_it, out_dims, &*vec1_it, vec1_dims, &*vec2_it, vec2_dims, vec_size);
		cudaStreamSynchronize(stream);
	}

	static void OuterProduct(iterator out_it, dim3 out_dims, iterator vec1_it, dim3 vec1_dims, size_t vec1_size, iterator vec2_it, dim3 vec2_dims, size_t vec2_size)
	{
		OuterHost<T, decltype(Product)*, Product>(&*out_it, out_dims, &*vec1_it, vec1_dims, vec1_size, &*vec2_it, vec2_dims, vec2_size);
	}

	static void OuterProduct(iterator out_it, dim3 out_dims, iterator vec1_it, dim3 vec1_dims, size_t vec1_size, iterator vec2_it, dim3 vec2_dims, size_t vec2_size, cudaStream_t stream)
	{
		dim3 dims = dims_max(out_dims, vec1_dims, vec2_dims);
		size_t threads = 1024 < vec2_size ? 1024 : vec2_size;
		OuterKernel<T, decltype(Product)*, Product><<<dims, threads, 0, stream>>>(&*out_it, out_dims, &*vec1_it, vec1_dims, vec1_size, &*vec2_it, vec2_dims, vec2_size);
		cudaStreamSynchronize(stream);
	}

	static void BroadcastProduct(iterator out_it, dim3 out_dims, iterator vec_it, dim3 vec_dims, size_t vec_size, iterator val_it, dim3 val_dims)
	{
		BroadcastHost<T, decltype(Product)*, Product>(&*out_it, out_dims, &*vec_it, vec_dims, vec_size, &*val_it, val_dims);
	}

	static void BroadcastProduct(iterator out_it, dim3 out_dims, iterator vec_it, dim3 vec_dims, size_t vec_size, iterator val_it, dim3 val_dims, cudaStream_t stream)
	{
		dim3 dims = dims_max(out_dims, vec_dims, val_dims);
		size_t threads = 1024 < vec_size ? 1024 : vec_size;
		BroadcastKernel<T, decltype(Product)*, Product><<<dims, threads, 0, stream>>>(&*out_it, out_dims, &*vec_it, vec_dims, vec_size, &*val_it, val_dims);
		cudaStreamSynchronize(stream);
	}
};
