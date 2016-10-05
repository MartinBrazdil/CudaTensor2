#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>

#include "Tensor.cu"

class MNIST
{
public:
	struct InputAndTarget { Tensor<> input, target; };
	using Iterator = std::vector<InputAndTarget>::iterator;
	struct Batch { Iterator begin, end; };

	const size_t label_count = 10;
	size_t train_count, test_count;
	size_t all_count = train_count + test_count;

	std::vector<InputAndTarget> data;

	MNIST(std::string dir_path, size_t width, size_t height, int load_train_count, int load_test_count, bool noise, bool normalize)
		: train_count(std::min(load_train_count, 60000)), test_count(std::min(load_test_count, 10000))
	{
		std::string trainImagesPath = dir_path + std::string("train-images.idx3-ubyte");
		std::string trainLabelsPath = dir_path + std::string("train-labels.idx1-ubyte");
		std::string testImagesPath = dir_path + std::string("t10k-images.idx3-ubyte");
		std::string testLabelsPath = dir_path + std::string("t10k-labels.idx1-ubyte");

		LoadData(trainImagesPath, trainLabelsPath, data, width, height, train_count, noise, normalize);
		LoadData(testImagesPath, testLabelsPath, data, width, height, test_count, noise, normalize);
	}

	void LoadData(std::string imagesPath, std::string labelsPath, std::vector<InputAndTarget>& data, size_t width, size_t height, size_t count, bool noise, bool normalize)
	{
		std::mt19937 rng{unsigned int(std::chrono::system_clock::now().time_since_epoch().count())};
		std::normal_distribution<double> normal(0, 0.01);

		// load training images
		std::ifstream trainImages(imagesPath, std::ios::binary);
		int32_t image_magic = LoadBytesSwapEndian(trainImages);
		int32_t image_total = LoadBytesSwapEndian(trainImages);
		int32_t image_rowCnt = LoadBytesSwapEndian(trainImages);
		int32_t image_colCnt = LoadBytesSwapEndian(trainImages);
		count = image_total < count ? image_total : count;
		int size = image_colCnt * image_rowCnt;
		char* raw_buffer = new char[image_colCnt * image_rowCnt];

		// load training labels
		std::ifstream trainLabels(labelsPath, std::ios::binary);
		int32_t label_magic = LoadBytesSwapEndian(trainLabels);
		int32_t label_total = LoadBytesSwapEndian(trainLabels);
		count = label_total < count ? label_total : count;
		char raw_label;

		data.reserve(count);
		for (int i = 0; i < count; i++)
		{
			Tensor<> image({width, height, 1});
			trainImages.read(raw_buffer, size);
			for (int j = 0; j < size; j++)
			{
				uint8_t p = static_cast<uint8_t>(raw_buffer[j]);
				image.memory[j] = p / 255.0;

				if (noise)
					image.memory[j] += normal(rng);
			}
			if (normalize)
				image.NormalizeToInterval(0, 1);

			trainLabels.read(&raw_label, 1);
			Tensor<> label({label_count});
			label.memory[static_cast<uint8_t>(raw_label)] = 1;

			data.push_back({image, label});
		}
		delete[] raw_buffer;
		trainImages.close();
		trainLabels.close();
	}

	int32_t LoadBytesSwapEndian(std::ifstream& stream)
	{
		char bytes[4];
		for (int i = 3; i >= 0; --i)
			stream.get(bytes[i]);
		return *(int32_t*)bytes;
	}

	Batch XValidationTestBatch(size_t factor, size_t idx)
	{
		Batch test_batch;
		size_t ration = all_count / factor;
		test_batch.begin = data.begin() + idx * ration;
		test_batch.end = test_batch.begin + ration;
		return test_batch;
	}
};
