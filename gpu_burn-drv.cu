/*
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

#include <cstdio>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <memory>
#include <random>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>
#include <nvml.h>
#include "cublas_v2.h"

#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

constexpr std::size_t SIZE{ 2048ul }; // Matrices are SIZE*SIZE..  2048^2 should be efficiently implemented in CUBLAS
constexpr double USEMEM{ 0.9 }; // Try to allocate 90% of memory
constexpr std::chrono::seconds TEMP_TIMEOUT{ 4u };	// Timeout between temperature readings

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
constexpr std::size_t OPS_PER_MUL{ 17188257792ul };

 // Actually, there are no rounding errors due to results being accumulated in an arbitrary order..
 // Therefore EPSILON = 0.0f is OK
template < class T >
struct Constants {};
template <>
struct Constants<float> {
	static constexpr float EPSILON{ 0.001f };
};
template <>
struct Constants<double> {
	static constexpr double EPSILON{ 0.0000001 };
};

template < class T >
__global__ void compare_kernel(T* C, int* faultyElems, size_t iters) {
	size_t iterStep = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	size_t myIndex = (blockIdx.y * blockDim.y + threadIdx.y) * // Y
		gridDim.x * blockDim.x + // W
		blockIdx.x * blockDim.x + threadIdx.x; // X

	int myFaulty = 0;
	for(size_t i = 1; i < iters; ++i)
		if(std::abs(C[myIndex] - C[myIndex + i * iterStep]) > Constants<T>::EPSILON)
			myFaulty++;

	::atomicAdd(faultyElems, myFaulty);
}

void checkError(cudaError_t rCode, std::string desc = "") {
	if(rCode != cudaSuccess) {
		static std::string error = cudaGetErrorName(rCode);
		throw ((desc == "") ?
			   std::string("Error: ") :
			   (std::string("Error in \"") + desc + std::string("\": "))) + error;
	}
}

void checkError(cublasStatus_t rCode, std::string desc = "") {
	static std::map<cublasStatus_t, std::string> g_errorStrings;
	if(!g_errorStrings.size()) {
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED"));
		g_errorStrings.insert(std::pair<cublasStatus_t, std::string>(CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR"));
	}

	if(rCode != CUBLAS_STATUS_SUCCESS)
		throw ((desc == "") ?
			   std::string("Error: ") :
			   (std::string("Error in \"") + desc + std::string("\": "))) +
		g_errorStrings[rCode];
}

std::atomic_bool g_running{ false };

template <class T> class GPU_Test {
public:
	GPU_Test(int dev, bool doubles, bool tensors) :
		d_devNumber(dev), d_doubles(doubles), d_tensors(tensors) {
		checkError(cudaSetDevice(dev), "Set thread device");
		//checkError(cublasInit());
		checkError(cublasCreate(&d_cublas), "init");

		if(d_tensors)
			checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));

		checkError(cudaMallocHost(&d_faultyElemsHost, sizeof(int)));
		d_error = 0;

		g_running.store(true);
	}
	~GPU_Test() {
		checkError(cudaFree(d_Cdata), "Free A");
		checkError(cudaFree(d_Adata), "Free B");
		checkError(cudaFree(d_Bdata), "Free C");
		cudaFreeHost(d_faultyElemsHost);
		printf("Freed memory for dev %d\n", d_devNumber);

		cublasDestroy(d_cublas);
		printf("Uninitted cublas\n");
	}

	static void termHandler(int signum) {
		g_running = false;
	}

	unsigned long long int getErrors() {
		if(*d_faultyElemsHost) {
			d_error += static_cast<long long int>(*d_faultyElemsHost);
		}
		unsigned long long int tempErrs = d_error;
		d_error = 0;
		return tempErrs;
	}

	size_t getIters() {
		return d_iters;
	}

	size_t totalMemory() {
		size_t freeMem, totalMem;
		checkError(cudaMemGetInfo(&freeMem, &totalMem));
		return totalMem;
	}

	size_t availMemory() {
		size_t freeMem, totalMem;
		checkError(cudaMemGetInfo(&freeMem, &totalMem));
		return freeMem;
	}

	void initBuffers(T* A, T* B) {
		const std::size_t useBytes = static_cast<size_t>(static_cast<double>(availMemory()) * USEMEM);
		cudaDeviceProp props;
		checkError(cudaGetDeviceProperties(&props, d_devNumber), "device properties");
		printf("Initialized device %d (%s) with %zu MB of memory (%zu MB available, using %zu MB of it), %s%s\n",
			   d_devNumber, props.name, totalMemory() / 1024ul / 1024ul, availMemory() / 1024ul / 1024ul, useBytes / 1024ul / 1024ul,
			   d_doubles ? "using DOUBLES" : "using FLOATS", d_tensors ? ", using Tensor Cores" : "");
		const std::size_t d_resultSize = sizeof(T) * SIZE * SIZE;
		d_iters = (useBytes - 2 * d_resultSize) / d_resultSize; // We remove A and B sizes
		//printf("Results are %d bytes each, thus performing %d iterations\n", d_resultSize, d_iters);
		checkError(cudaMalloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
		checkError(cudaMalloc(&d_Adata, d_resultSize), "A alloc");
		checkError(cudaMalloc(&d_Bdata, d_resultSize), "B alloc");

		checkError(cudaMalloc(&d_faultyElemData, sizeof(int)), "faulty data");

		// Populating matrices A and B
		checkError(cudaMemcpy(d_Adata, A, d_resultSize, cudaMemcpyHostToDevice), "A -> device");
		checkError(cudaMemcpy(d_Bdata, B, d_resultSize, cudaMemcpyHostToDevice), "A -> device");

		initCompareKernel();
	}

	void compute() {
		static const float alpha = 1.0f;
		static const float beta = 0.0f;
		static const double alphaD = 1.0;
		static const double betaD = 0.0;

		for(size_t i = 0; i < d_iters; ++i) {
			if(d_doubles)
				checkError(cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
									   SIZE, SIZE, SIZE, &alphaD,
									   (const double*)d_Adata, SIZE,
									   (const double*)d_Bdata, SIZE,
									   &betaD,
									   (double*)d_Cdata + i * SIZE * SIZE, SIZE), "DGEMM");
			else
				checkError(cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
									   SIZE, SIZE, SIZE, &alpha,
									   (const float*)d_Adata, SIZE,
									   (const float*)d_Bdata, SIZE,
									   &beta,
									   (float*)d_Cdata + i * SIZE * SIZE, SIZE), "SGEMM");
		}
	}

	void initCompareKernel() {
		checkError(cudaFuncSetCacheConfig(compare_kernel<T>, cudaFuncCachePreferL1), "L1 config");
	}

	void compare() {
		checkError(cudaMemset(d_faultyElemData, 0, 4), "memset");
		const dim3 gridSize{ SIZE / g_blockSize, SIZE / g_blockSize, 1u };
		const dim3 blockSize{ g_blockSize, g_blockSize, 1u };
		compare_kernel<T><<<gridSize, blockSize>>>(d_Cdata, d_faultyElemData, d_iters);
		checkError(cudaMemcpy(d_faultyElemsHost, d_faultyElemData, sizeof(int), cudaMemcpyDeviceToHost), "Read faultyelemdata");
	}

	bool shouldRun() {
		return g_running.load(std::memory_order_acquire);
	}

private:
	bool d_doubles;
	bool d_tensors;
	int d_devNumber;
	size_t d_iters;
	size_t d_resultSize;

	long long int d_error;

	static const int g_blockSize = 16;


	T* d_Cdata;
	T* d_Adata;
	T* d_Bdata;
	int* d_faultyElemData;
	int* d_faultyElemsHost;

	cublasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
	int deviceCount = 0;
	checkError(cudaGetDeviceCount(&deviceCount));

	if(!deviceCount)
		throw std::string("No CUDA devices");

#ifdef USEDEV
	if(USEDEV >= deviceCount)
		throw std::string("Not enough devices for USEDEV");
#endif

	return deviceCount;
}

template < class T >
void startBurn(int index, T* A, T* B, bool doubles, bool tensors) {
	std::unique_ptr<GPU_Test<T>> our;
	try {
		our = std::make_unique<GPU_Test<T>>(index, doubles, tensors);
		our->initBuffers(A, B);
	} catch(std::string e) {
		fprintf(stderr, "Couldn't init a GPU test: %s\n", e.c_str());
		exit(124);
	}

	// The actual work
	try {
		int eventIndex = 0;
		const int maxEvents = 2;
		cudaEvent_t events[maxEvents];
		for(int i = 0; i < maxEvents; ++i)
			cudaEventCreateWithFlags(events + i, cudaEventDefault);

		int nonWorkIters = maxEvents;

		auto timestamp = std::chrono::high_resolution_clock::now();

		while(our->shouldRun()) {
			our->compute();
			our->compare();
			checkError(cudaEventRecord(events[eventIndex]), "Record event");

			eventIndex = ++eventIndex % maxEvents;

			//while(cudaEventQuery(events[eventIndex]) != cudaSuccess)
			cudaEventSynchronize(events[eventIndex]);

			if(--nonWorkIters > 0) continue;

			const auto processed = static_cast<std::size_t>(our->getIters());
			const auto flops = static_cast<double>(processed) * static_cast<double>(OPS_PER_MUL);
			const auto currStamp = std::chrono::high_resolution_clock::now();
			const auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(currStamp - timestamp);
			timestamp = currStamp;
			const auto gFlops = (static_cast<double>(processed * OPS_PER_MUL) / delta.count()) / 1'000'000'000.0;

			printf("Device %d: %zu iterations (%f GFLOP/s)\n", index, processed, gFlops);
		}

		for(int i = 0; i < maxEvents; ++i)
			cudaEventSynchronize(events[i]);
	} catch(std::string e) {
		fprintf(stderr, "Failure during compute: %s\n", e.c_str());
		exit(111);
	}
}


template < class T >
void launch(std::chrono::seconds runLength, bool useDoubles, bool useTensorCores) {
	// Initting A and B with random data
	auto A = std::make_unique<T[]>(SIZE * SIZE);
	auto B = std::make_unique<T[]>(SIZE * SIZE);
	std::mt19937_64 engine(std::random_device{}());
	std::uniform_real_distribution<T> dist;
	for(size_t i = 0; i < SIZE * SIZE; ++i) {
		A[i] = dist(engine);
		B[i] = dist(engine);
	}

	const auto deviceCount = initCuda();
	if(deviceCount <= 0) {
		fprintf(stderr, "No CUDA devices\n");
		exit(EXIT_FAILURE);
	}

	std::vector<nvmlDevice_t> deviceHandles(static_cast<std::size_t>(deviceCount));
	for(int d = 0; d < deviceCount; ++d)
		::nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(d), &deviceHandles[d]);

	// TODO: replace with condition variable?
	g_running.store(true, std::memory_order_release);
	// One thread per device as well as one for temperature updates
	std::vector<std::thread> threads;
	threads.reserve(static_cast<std::size_t>(deviceCount) + 1u);
	threads.emplace_back([&deviceHandles]() {
		while(g_running.load(std::memory_order_acquire)) {
			std::this_thread::sleep_for(TEMP_TIMEOUT);
			printf("Device temps:");
			for(std::size_t i = 0u; i < deviceHandles.size(); ++i) {
				unsigned temp = 0u;
				nvmlDeviceGetTemperature(deviceHandles[i], NVML_TEMPERATURE_GPU, &temp);
				printf(" %u", temp);
			}
			printf(" (in deg. C)\n");
		}
	});

	for(int d = 0; d < deviceCount; ++d)
		threads.emplace_back(startBurn<T>, d, A.get(), B.get(), useDoubles, useTensorCores);

	std::this_thread::sleep_for(runLength);
	g_running.store(false, std::memory_order_release);

	for(std::size_t i = 1u; i < threads.size(); ++i) {
		if(threads[i].joinable())
			threads[i].join();
	}
	if(threads[0].joinable())
		threads[0].join();
}

int main(int argc, char** argv) {
	std::chrono::seconds runLength{ 10 };
	bool useDoubles = false;
	bool useTensorCores = false;
	int thisParam = 0;

	std::vector<std::string> args(argv, argv + argc);
	for(size_t i = 1; i < args.size(); ++i) {
		if(argc >= 2 && std::string(argv[i]).find("-d") != std::string::npos) {
			useDoubles = true;
			thisParam++;
		}
		if(argc >= 2 && std::string(argv[i]).find("-tc") != std::string::npos) {
			useTensorCores = true;
			thisParam++;
		}
	}

	if(argc - thisParam < 2)
		printf("Run length not specified in the command line.  Burning for 10 secs\n");
	else
		runLength = std::chrono::seconds{ atoi(argv[1 + thisParam]) };

	const auto res = ::nvmlInit_v2();
	if(res != NVML_SUCCESS) {
		switch(res) {
			case NVML_ERROR_DRIVER_NOT_LOADED: fprintf(stderr, "Nvidia driver is not running\n"); break;
			case NVML_ERROR_NO_PERMISSION: fprintf(stderr, "NVML does not have permission to talk to the driver\n"); break;
			default: fprintf(stderr, "Unknown error\n"); break;
		}
		return EXIT_FAILURE;
	}

	if(useDoubles)
		launch<double>(runLength, useDoubles, useTensorCores);
	else
		launch<float>(runLength, useDoubles, useTensorCores);

	::nvmlShutdown();

	return 0;
}