#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10
#define THREADCOUNT 128  

// Timer function (CPU-based)
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate and free matrix (host-side)
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}

// CUDA error checking macro
#define CUDA_CHECK(err) do { \
    cudaError_t err_code = (err); \
    if (err_code != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err_code)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(err) do { \
    cublasStatus_t err_code = (err); \
    if (err_code != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, err_code); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA kernels
__global__ void reluActivationKernel(double* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__global__ void reluActivationFloatKernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = (input[idx] > 0) ? input[idx] : 0;
    }
}

__global__ void backwardOutputGradKernel(double* output, double* target, double* d_output, int batchSize) {
    int i = threadIdx.x;
    int sample = blockIdx.x;

    if (sample < batchSize && i < OUTPUT_SIZE) {
        d_output[sample * OUTPUT_SIZE + i] = output[sample * OUTPUT_SIZE + i] - target[sample * OUTPUT_SIZE + i];
    }
}

__global__ void backwardHiddenGradKernel(double* hidden, double* d_hidden, int batchSize, int hiddenSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = idx / hiddenSize;
    int neuron = idx % hiddenSize;
   
    if (sample < batchSize && neuron < hiddenSize) {
        int index = sample * hiddenSize + neuron;
        d_hidden[index] = d_hidden[index] * (hidden[index] > 0 ? 1.0 : 0.0);
    }
}

__global__ void addBiasKernel(float* output, float* bias, int batchSize, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neuron = idx % size;
    int sample = idx / size;
   
    if (sample < batchSize && neuron < size) {
        output[sample * size + neuron] += bias[neuron];
    }
}

__global__ void softmaxKernel(double* output, int batchSize, int size) {
    int sample = blockIdx.x;
   
    if (sample < batchSize) {
        double max_val = output[sample * size];
        for (int i = 1; i < size; i++) {
            if (output[sample * size + i] > max_val) {
                max_val = output[sample * size + i];
            }
        }
       
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            output[sample * size + i] = exp(output[sample * size + i] - max_val);
            sum += output[sample * size + i];
        }

        for (int i = 0; i < size; i++) {
            output[sample * size + i] /= sum;
        }
    }
}

// Kernel to convert from double to float
__global__ void doubleToFloatKernel(double* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (float)input[idx];
    }
}

// Kernel to convert from float to double
__global__ void floatToDoubleKernel(float* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (double)input[idx];
    }
}

// Neural network structure
typedef struct {
    double *W1, *W2, *b1, *b2; // Device pointers (double precision)
    double **W1_host, **W2_host; // Host pointers for initialization
    cublasHandle_t cublas_handle; // cuBLAS handle
    float *W1_float, *W2_float; // Float versions for tensor cores
    float *b1_float, *b2_float; // Float versions of biases
    float *input_float, *hidden_float, *output_float; // Float buffers for computations
    float *d_hidden_float, *d_output_float; // Float buffers for gradients
} NeuralNetwork;

// Initialize network (host and device)
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1_host = allocateMatrix(INPUT_SIZE, HIDDEN_SIZE); // Transpose for cuBLAS
    net->W2_host = allocateMatrix(HIDDEN_SIZE, OUTPUT_SIZE); // Transpose for cuBLAS

    srand(time(NULL));
    // Initialize weights with small random values
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W1_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < OUTPUT_SIZE; j++)
            net->W2_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Allocate device memory for double precision
    CUDA_CHECK(cudaMalloc(&net->W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b2, OUTPUT_SIZE * sizeof(double)));

    // Allocate device memory for single precision (for tensor cores)
    CUDA_CHECK(cudaMalloc(&net->W1_float, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->W2_float, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b1_float, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->b2_float, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->input_float, BATCH_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->hidden_float, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->output_float, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_hidden_float, BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->d_output_float, BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Copy weights to device (flattened for cuBLAS column-major format)
    double* temp_W1 = (double*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* temp_W2 = (double*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
   
    // Flatten W1 in column-major order for cuBLAS
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            temp_W1[j * INPUT_SIZE + i] = net->W1_host[i][j];
        }
    }
   
    // Flatten W2 in column-major order for cuBLAS
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            temp_W2[j * HIDDEN_SIZE + i] = net->W2_host[i][j];
        }
    }
   
    CUDA_CHECK(cudaMemcpy(net->W1, temp_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(net->W2, temp_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
   
    free(temp_W1);
    free(temp_W2);

    // Zero-initialize biases
    CUDA_CHECK(cudaMemset(net->b1, 0, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(net->b2, 0, OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(net->b1_float, 0, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(net->b2_float, 0, OUTPUT_SIZE * sizeof(float)));

    // Convert weights to float for tensor cores
    int blockSize = 256;
    int gridSize = (INPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;
    doubleToFloatKernel<<<gridSize, blockSize>>>(net->W1, net->W1_float, INPUT_SIZE * HIDDEN_SIZE);
   
    gridSize = (HIDDEN_SIZE * OUTPUT_SIZE + blockSize - 1) / blockSize;
    doubleToFloatKernel<<<gridSize, blockSize>>>(net->W2, net->W2_float, HIDDEN_SIZE * OUTPUT_SIZE);
   
    // Convert biases to float
    gridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
    doubleToFloatKernel<<<gridSize, blockSize>>>(net->b1, net->b1_float, HIDDEN_SIZE);
   
    gridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
    doubleToFloatKernel<<<gridSize, blockSize>>>(net->b2, net->b2_float, OUTPUT_SIZE);
   
    CUDA_CHECK(cudaDeviceSynchronize());

    // Initialize cuBLAS
    CUBLAS_CHECK(cublasCreate(&net->cublas_handle));
   
    return net;
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double *input_batch, *hidden_batch, *output_batch, *target_batch, *d_output, *d_hidden;

    CUDA_CHECK(cudaMalloc(&input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&target_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));

    // Constants for cuBLAS calls
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    float beta_bias_f = 1.0f;
   
    double alpha_d = 1.0;
    double beta_d = 0.0;
   
    int blockSize = 256;

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int actualBatchSize = (i + BATCH_SIZE > numImages) ? (numImages - i) : BATCH_SIZE;

            // Prepare input and target data
            double* host_input = (double*)malloc(actualBatchSize * INPUT_SIZE * sizeof(double));
            double* host_target = (double*)malloc(actualBatchSize * OUTPUT_SIZE * sizeof(double));
            double* host_output = (double*)malloc(actualBatchSize * OUTPUT_SIZE * sizeof(double));

            for (int b = 0; b < actualBatchSize; b++) {
                for (int j = 0; j < INPUT_SIZE; j++)
                    host_input[b * INPUT_SIZE + j] = images[i + b][j];
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    host_target[b * OUTPUT_SIZE + j] = labels[i + b][j];
            }

            // Copy input to device
            CUDA_CHECK(cudaMemcpy(input_batch, host_input, actualBatchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(target_batch, host_target, actualBatchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
           
            // Convert input to float
            int gridSize = (actualBatchSize * INPUT_SIZE + blockSize - 1) / blockSize;
            doubleToFloatKernel<<<gridSize, blockSize>>>(input_batch, net->input_float, actualBatchSize * INPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     HIDDEN_SIZE, actualBatchSize, INPUT_SIZE,
                                     &alpha_f,
                                     net->W1_float, CUDA_R_32F, HIDDEN_SIZE,
                                     net->input_float, CUDA_R_32F, INPUT_SIZE,
                                     &beta_f,
                                     net->hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                     CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
           
            gridSize = (HIDDEN_SIZE * actualBatchSize + blockSize - 1) / blockSize;
            addBiasKernel<<<gridSize, blockSize>>>(net->hidden_float, net->b1_float, actualBatchSize, HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
            reluActivationFloatKernel<<<gridSize, blockSize>>>(net->hidden_float, actualBatchSize * HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Forward pass output layer using cuBLAS (output = hidden * W2^T + b2)
            CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     OUTPUT_SIZE, actualBatchSize, HIDDEN_SIZE,
                                     &alpha_f,
                                     net->W2_float, CUDA_R_32F, OUTPUT_SIZE,
                                     net->hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                     &beta_f,
                                     net->output_float, CUDA_R_32F, OUTPUT_SIZE,
                                     CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
           
            // Add bias to output layer
            gridSize = (OUTPUT_SIZE * actualBatchSize + blockSize - 1) / blockSize;
            addBiasKernel<<<gridSize, blockSize>>>(net->output_float, net->b2_float, actualBatchSize, OUTPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Convert output back to double for softmax
            gridSize = (actualBatchSize * OUTPUT_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->output_float, output_batch, actualBatchSize * OUTPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Softmax activation
            softmaxKernel<<<actualBatchSize, 1>>>(output_batch, actualBatchSize, OUTPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy output to host for loss calculation
            CUDA_CHECK(cudaMemcpy(host_output, output_batch, actualBatchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));


            backwardOutputGradKernel<<<actualBatchSize, OUTPUT_SIZE>>>(output_batch, target_batch, d_output, actualBatchSize);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->hidden_float, hidden_batch, actualBatchSize * HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Convert d_output to float
            gridSize = (actualBatchSize * OUTPUT_SIZE + blockSize - 1) / blockSize;
            doubleToFloatKernel<<<gridSize, blockSize>>>(d_output, net->d_output_float, actualBatchSize * OUTPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Backward pass hidden gradient using cuBLAS (d_hidden = W2 * d_output)
            CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                     CUBLAS_OP_T, CUBLAS_OP_N,
                                     HIDDEN_SIZE, actualBatchSize, OUTPUT_SIZE,
                                     &alpha_f,
                                     net->W2_float, CUDA_R_32F, OUTPUT_SIZE,
                                     net->d_output_float, CUDA_R_32F, OUTPUT_SIZE,
                                     &beta_f,
                                     net->d_hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                     CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
           
            // Convert d_hidden to double
            gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->d_hidden_float, d_hidden, actualBatchSize * HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Apply ReLU gradient
            gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
            backwardHiddenGradKernel<<<gridSize, blockSize>>>(hidden_batch, d_hidden, actualBatchSize, HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
            doubleToFloatKernel<<<gridSize, blockSize>>>(d_hidden, net->d_hidden_float, actualBatchSize * HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
           
            // Update W2 using cuBLAS (W2 -= lr * d_output * hidden^T / batch_size)
            float lr_over_batch = -LEARNING_RATE / actualBatchSize;
            CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     OUTPUT_SIZE, HIDDEN_SIZE, actualBatchSize,
                                     &lr_over_batch,
                                     net->d_output_float, CUDA_R_32F, OUTPUT_SIZE,
                                     net->hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                     &alpha_f,
                                     net->W2_float, CUDA_R_32F, OUTPUT_SIZE,
                                     CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
           
            // Update W1 using cuBLAS (W1 -= lr * d_hidden * input^T / batch_size)
            CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                     CUBLAS_OP_N, CUBLAS_OP_T,
                                     HIDDEN_SIZE, INPUT_SIZE, actualBatchSize,
                                     &lr_over_batch,
                                     net->d_hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                     net->input_float, CUDA_R_32F, INPUT_SIZE,
                                     &alpha_f,
                                     net->W1_float, CUDA_R_32F, HIDDEN_SIZE,
                                     CUBLAS_COMPUTE_32F_FAST_TF32,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
           
            float* host_d_output_float = (float*)malloc(actualBatchSize * OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(host_d_output_float, net->d_output_float, actualBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
           
            float* host_b2_float = (float*)malloc(OUTPUT_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(host_b2_float, net->b2_float, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
           
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float sum = 0.0f;
                for (int b = 0; b < actualBatchSize; b++) {
                    sum += host_d_output_float[b * OUTPUT_SIZE + j];
                }
                host_b2_float[j] -= LEARNING_RATE * sum / actualBatchSize;
            }
           
            CUDA_CHECK(cudaMemcpy(net->b2_float, host_b2_float, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            free(host_b2_float);
            free(host_d_output_float);
           
            // Update biases (b1 -= lr * sum(d_hidden) / batch_size)
            float* host_d_hidden_float = (float*)malloc(actualBatchSize * HIDDEN_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(host_d_hidden_float, net->d_hidden_float, actualBatchSize * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
           
            float* host_b1_float = (float*)malloc(HIDDEN_SIZE * sizeof(float));
            CUDA_CHECK(cudaMemcpy(host_b1_float, net->b1_float, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
           
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float sum = 0.0f;
                for (int b = 0; b < actualBatchSize; b++) {
                    sum += host_d_hidden_float[b * HIDDEN_SIZE + j];
                }
                host_b1_float[j] -= LEARNING_RATE * sum / actualBatchSize;
            }
           
            CUDA_CHECK(cudaMemcpy(net->b1_float, host_b1_float, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
            free(host_b1_float);
            free(host_d_hidden_float);
           
            // Convert updated weights and biases back to double precision
            gridSize = (INPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->W1_float, net->W1, HIDDEN_SIZE * INPUT_SIZE);
           
            gridSize = (OUTPUT_SIZE * HIDDEN_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->W2_float, net->W2, OUTPUT_SIZE * HIDDEN_SIZE);
           
            gridSize = (HIDDEN_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->b1_float, net->b1, HIDDEN_SIZE);
           
            gridSize = (OUTPUT_SIZE + blockSize - 1) / blockSize;
            floatToDoubleKernel<<<gridSize, blockSize>>>(net->b2_float, net->b2, OUTPUT_SIZE);
           
            CUDA_CHECK(cudaDeviceSynchronize());

            // Calculate loss and accuracy
            for (int b = 0; b < actualBatchSize; b++) {
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (host_output[b * OUTPUT_SIZE + j] > host_output[b * OUTPUT_SIZE + pred])
                        pred = j;
                    if (labels[i + b][j] > labels[i + b][actual])
                        actual = j;
                    loss -= labels[i + b][j] * log(host_output[b * OUTPUT_SIZE + j] + 1e-8);
                }
                if (pred == actual) correct++;
            }
           
            free(host_input);
            free(host_target);
            free(host_output);
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
   
    cudaFree(input_batch);
    cudaFree(hidden_batch);
    cudaFree(output_batch);
    cudaFree(target_batch);
    cudaFree(d_output);
    cudaFree(d_hidden);
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double *input_batch, *output_batch, *hidden_batch;
   
    CUDA_CHECK(cudaMalloc(&input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));

    // Constants for cuBLAS calls
    float alpha_f = 1.0f;
    float beta_f = 0.0f;
   
    int blockSize = 256;

    int correct = 0;
    clock_t eval_start = clock();

    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        int actualBatchSize = (i + BATCH_SIZE > numImages) ? (numImages - i) : BATCH_SIZE;

        double* host_input = (double*)malloc(actualBatchSize * INPUT_SIZE * sizeof(double));
        double* host_output = (double*)malloc(actualBatchSize * OUTPUT_SIZE * sizeof(double));

        for (int b = 0; b < actualBatchSize; b++) {
            for (int j = 0; j < INPUT_SIZE; j++)
                host_input[b * INPUT_SIZE + j] = images[i + b][j];
        }

        CUDA_CHECK(cudaMemcpy(input_batch, host_input, actualBatchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

        // Convert input to float
        int gridSize = (actualBatchSize * INPUT_SIZE + blockSize - 1) / blockSize;
        doubleToFloatKernel<<<gridSize, blockSize>>>(input_batch, net->input_float, actualBatchSize * INPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
       
        // Forward pass hidden layer
        CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 HIDDEN_SIZE, actualBatchSize, INPUT_SIZE,
                                 &alpha_f,
                                 net->W1_float, CUDA_R_32F, HIDDEN_SIZE,
                                 net->input_float, CUDA_R_32F, INPUT_SIZE,
                                 &beta_f,
                                 net->hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                 CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
       
        // Add bias to hidden layer - using the same kernel as in train()
        gridSize = (HIDDEN_SIZE * actualBatchSize + blockSize - 1) / blockSize;
        addBiasKernel<<<gridSize, blockSize>>>(net->hidden_float, net->b1_float, actualBatchSize, HIDDEN_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
       
        // Apply ReLU activation directly on float data
        gridSize = (actualBatchSize * HIDDEN_SIZE + blockSize - 1) / blockSize;
        reluActivationFloatKernel<<<gridSize, blockSize>>>(net->hidden_float, actualBatchSize * HIDDEN_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
       
        // Forward pass output layer
        CUBLAS_CHECK(cublasGemmEx(net->cublas_handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 OUTPUT_SIZE, actualBatchSize, HIDDEN_SIZE,
                                 &alpha_f,
                                 net->W2_float, CUDA_R_32F, OUTPUT_SIZE,
                                 net->hidden_float, CUDA_R_32F, HIDDEN_SIZE,
                                 &beta_f,
                                 net->output_float, CUDA_R_32F, OUTPUT_SIZE,
                                 CUBLAS_COMPUTE_32F_FAST_TF32,
                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP));
       
        // Add bias to output layer
        gridSize = (OUTPUT_SIZE * actualBatchSize + blockSize - 1) / blockSize;
        addBiasKernel<<<gridSize, blockSize>>>(net->output_float, net->b2_float, actualBatchSize, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
       
        // Convert output to double for softmax
        gridSize = (actualBatchSize * OUTPUT_SIZE + blockSize - 1) / blockSize;
        floatToDoubleKernel<<<gridSize, blockSize>>>(net->output_float, output_batch, actualBatchSize * OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
       
        // Softmax
        softmaxKernel<<<actualBatchSize, 1>>>(output_batch, actualBatchSize, OUTPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(host_output, output_batch, actualBatchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        for (int b = 0; b < actualBatchSize; b++) {
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (host_output[b * OUTPUT_SIZE + j] > host_output[b * OUTPUT_SIZE + pred])
                    pred = j;
                if (labels[i + b][j] > labels[i + b][actual])
                    actual = j;
            }
            if (pred == actual) correct++;
        }
        
        free(host_input);
        free(host_output);
    }

    double accuracy = (double)correct / numImages * 100.0;
    printf("Evaluation Accuracy: %.2f%% - Time: %.3fs\n", accuracy, get_time(eval_start));

    cudaFree(input_batch);
    cudaFree(output_batch);
    cudaFree(hidden_batch);
}

// MNIST loading and freeing unchanged
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("Error opening %s\n", filename); exit(1); }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) { printf("Error opening %s\n", filename); exit(1); }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1_host, HIDDEN_SIZE);
    freeMatrix(net->W2_host, OUTPUT_SIZE);
    CUDA_CHECK(cudaFree(net->W1));
    CUDA_CHECK(cudaFree(net->W2));
    CUDA_CHECK(cudaFree(net->b1));
    CUDA_CHECK(cudaFree(net->b2));
   
    // Free the float version buffers
    CUDA_CHECK(cudaFree(net->W1_float));
    CUDA_CHECK(cudaFree(net->W2_float));
    CUDA_CHECK(cudaFree(net->input_float));
    CUDA_CHECK(cudaFree(net->hidden_float));
    CUDA_CHECK(cudaFree(net->output_float));
    CUDA_CHECK(cudaFree(net->d_hidden_float));
    CUDA_CHECK(cudaFree(net->d_output_float));
   
    // Destroy cuBLAS handle
    CUBLAS_CHECK(cublasDestroy(net->cublas_handle));
   
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network with CUDA and cuBLAS\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}
