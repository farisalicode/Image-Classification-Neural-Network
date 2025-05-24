#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10
#define THREADCOUNT 128  
//cuz 128 are the max nodes working parallel

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
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)








// CUDA kernels
__global__ void forwardHiddenKernel(double* W1, double* b1, double* input, double* hidden, int inputSize) {

   //no reuse in weights in a block(for an image) so no point of keeping them in shared memory 
   //we can use the input image, cuz it is reused by all 128 neurons

    //every block has an image 
    int sample = blockIdx.y; //for which block?
    int i = blockIdx.x * blockDim.x + threadIdx.x; //what thread is it?

    __shared__ double input_shared[INPUT_SIZE];
   
    for (int j = threadIdx.x; j < inputSize; j += blockDim.x) 
    {
        input_shared[j] = input[sample*inputSize + j];
    } //even out of bound threads are loading


    __syncthreads();



    if (i < HIDDEN_SIZE) {
        double sum = b1[i];
        for (int j = 0; j < inputSize; j++) {
            sum += W1[i * inputSize + j] * input_shared[j];
        }
        hidden[sample * HIDDEN_SIZE + i] = (sum > 0) ? sum : 0;
    }
}

__global__ void forwardOutputKernel(double* W2, double* b2, double* hidden, double* output, int hiddenSize) {

   //will use shared memory for hidden layer here 
   //in a block there is no repeated usage of weights so no need of those



    int sample = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double hidden_shared[HIDDEN_SIZE];
    for (int j = threadIdx.x; j < hiddenSize; j += blockDim.x) {
        hidden_shared[j] = hidden[sample * hiddenSize + j];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE) {
        double sum = b2[i];
        for (int j = 0; j < hiddenSize; j++) {
            sum += W2[i * hiddenSize + j] * hidden_shared[j];
        }
        output[sample * OUTPUT_SIZE + i] = sum;
    }
}













//10 threads against each image
// for d_output = output - target
__global__ void backwardOutputGradKernel(double* output, double* target, double* d_output) {
    int sample = blockIdx.x;
    int i = threadIdx.x;

    if (i < OUTPUT_SIZE) {
        d_output[sample * OUTPUT_SIZE + i] = output[sample * OUTPUT_SIZE + i] - target[sample * OUTPUT_SIZE + i];
    }
}





//number of kernels against every image will be ... hidden size i guess
__global__ void backwardHiddenGradKernel(double* W2, double* d_output, double* hidden, double* d_hidden, int hiddenSize) {
    

    //doutput getting reused 

    int sample = blockIdx.y; //each image
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double d_output_shared[OUTPUT_SIZE];
   
    for (int j = threadIdx.x; j < OUTPUT_SIZE; j += blockDim.x) 
    {
        d_output_shared[j] = d_output[sample * OUTPUT_SIZE + j];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE) {
        double sum = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            sum += W2[j * hiddenSize + i] * d_output_shared[j];
        }
        d_hidden[sample * HIDDEN_SIZE + i] = sum * (hidden[sample * HIDDEN_SIZE + i] > 0);
    }
}


//128 threads for each output (10 blocks) thing so.... for each row and each column 
//basically for the whole W2 matrix
__global__ void updateW2Kernel(double* W2, double* b2, double* d_output, double* hidden, int hiddenSize) {

  //d_output and hidden getting reused in every block 
    __shared__ double d_output_shared[BATCH_SIZE];
    //all the threads in a block are reusing these values 

   
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j < BATCH_SIZE) {
        d_output_shared[j] = d_output[j * OUTPUT_SIZE + i];
    }
    
    __syncthreads(); // Make sure all threads finish loading before proceeding
    

    if (i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
        double grad = 0;
        for (int s = 0; s < BATCH_SIZE; s++) {
            grad += d_output_shared[s] * hidden[s * hiddenSize + j];
        }
        W2[i * hiddenSize + j] -= LEARNING_RATE * grad / BATCH_SIZE;

        if (j == 0) {
            double b_grad = 0;
            for (int s = 0; s < BATCH_SIZE; s++) {
                b_grad += d_output_shared[s];
            }
            b2[i] -= LEARNING_RATE * b_grad / BATCH_SIZE;
        }
    }
}

__global__ void updateW1Kernel(double* W1, double* b1, double* d_hidden, double* input, int inputSize) {


   //now here similarly d_hidden is being reused 
   __shared__ double d_hidden_shared[BATCH_SIZE];

    int i = blockIdx.x;
    int j = threadIdx.x;

    if (j < BATCH_SIZE) {
        d_hidden_shared[j] = d_hidden[j * HIDDEN_SIZE + i];
    }

   __syncthreads();
 

    if (i < HIDDEN_SIZE && j < INPUT_SIZE) {
        double grad = 0;
        for (int s = 0; s < BATCH_SIZE; s++) {
            grad += d_hidden_shared[s] * input[s * inputSize + j];
        }
        W1[i * inputSize + j] -= LEARNING_RATE * grad / BATCH_SIZE;

        if (j == 0) {
            double b_grad = 0;
            for (int s = 0; s < BATCH_SIZE; s++) {
                b_grad += d_hidden_shared[s];
            }
            b1[i] -= LEARNING_RATE * b_grad / BATCH_SIZE;
        }
    }
}


//one thread for an ouput layer, cuz just ten values so, one thread is sufficient
__global__ void softmaxKernel(double* output, int batchSize, int size) {
    int sample = blockIdx.y;
    
    if (sample < batchSize) {
        
        double sum = 0.0;
        double temp = 0;
        for (int i = 0; i < size; i++) {
            temp = exp(output[sample * size + i]);
            output[sample * size + i] = temp;
            sum += temp;
        }

        for (int i = 0; i < size; i++) {
            output[sample * size + i] /= sum;
        }
    }
}




// Neural network structure
typedef struct {
    double *W1, *W2, *b1, *b2; // Device pointers
    double **W1_host, **W2_host; // Host pointers for initialization



} NeuralNetwork;

// Initialize network (host and device)
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1_host = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2_host = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2_host[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b1, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->b2, OUTPUT_SIZE * sizeof(double)));

    // Copy weights to device
    for (int i = 0; i < HIDDEN_SIZE; i++)
        CUDA_CHECK(cudaMemcpy(net->W1 + i * INPUT_SIZE, net->W1_host[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    for (int i = 0; i < OUTPUT_SIZE; i++)
        CUDA_CHECK(cudaMemcpy(net->W2 + i * HIDDEN_SIZE, net->W2_host[i], HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    // Zero-initialize biases
    CUDA_CHECK(cudaMemset(net->b1, 0, HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMemset(net->b2, 0, OUTPUT_SIZE * sizeof(double)));

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

    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
                                        //idhar change kia ha increment
             
            int actualBatchSize = (i + BATCH_SIZE > numImages) ? (numImages - i) : BATCH_SIZE;

            double host_input[BATCH_SIZE * INPUT_SIZE];
            double host_target[BATCH_SIZE * OUTPUT_SIZE];
            double host_output[BATCH_SIZE * OUTPUT_SIZE];

            for (int b = 0; b < actualBatchSize; b++) {
                       //running each iteration for every image

                for (int j = 0; j < INPUT_SIZE; j++)
                    host_input[b * INPUT_SIZE + j] = images[i + b][j];
                for (int j = 0; j < OUTPUT_SIZE; j++)
                    host_target[b * OUTPUT_SIZE + j] = labels[i + b][j];
            }


            //giving all the images
            CUDA_CHECK(cudaMemcpy(input_batch, host_input, actualBatchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            //giving their answers here 
            CUDA_CHECK(cudaMemcpy(target_batch, host_target, actualBatchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
                             //formality, its actually 1 block for each image
            dim3 grid_hidden((HIDDEN_SIZE + THREADCOUNT-1) / THREADCOUNT, actualBatchSize);
            forwardHiddenKernel<<<grid_hidden, THREADCOUNT>>>(net->W1, net->b1, input_batch, hidden_batch, INPUT_SIZE);
            //relu inside the kernel
            
            CUDA_CHECK(cudaDeviceSynchronize());

            //only 10 threads needed though but may help for shared memory
            dim3 grid_output((OUTPUT_SIZE + THREADCOUNT-1) / THREADCOUNT, actualBatchSize);
            forwardOutputKernel<<<grid_output, THREADCOUNT>>>(net->W2, net->b2, hidden_batch, output_batch, HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            //fixed hamza's Softmax

            dim3 softmax_grid(1, actualBatchSize);
            softmaxKernel<<<softmax_grid, 1>>>(output_batch, actualBatchSize, OUTPUT_SIZE);

            CUDA_CHECK(cudaMemcpy(host_output, output_batch, actualBatchSize * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));




            //for backward now 



            //here updating hiddend/outputlyare weights
            backwardOutputGradKernel<<<actualBatchSize, OUTPUT_SIZE>>>(output_batch, target_batch, d_output);
            CUDA_CHECK(cudaDeviceSynchronize());



            dim3 grid_hidden_grad((HIDDEN_SIZE + THREADCOUNT-1) / THREADCOUNT, actualBatchSize);
            //try to keep minimum threacount greater than hidden size
            backwardHiddenGradKernel<<<grid_hidden_grad, HIDDEN_SIZE>>>(net->W2, d_output, hidden_batch, d_hidden, HIDDEN_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());






                             //10 blocks, with 128 threads each 
            updateW2Kernel<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net->W2, net->b2, d_output, hidden_batch, HIDDEN_SIZE);


            updateW1Kernel<<<HIDDEN_SIZE, INPUT_SIZE>>>(net->W1, net->b1, d_hidden, input_batch, INPUT_SIZE);
            CUDA_CHECK(cudaDeviceSynchronize());






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
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    double *input_batch, *hidden_batch, *output_batch;

    CUDA_CHECK(cudaMalloc(&input_batch, BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&hidden_batch, BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&output_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));

    int correct = 0;
    clock_t eval_start = clock();

    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        int actualBatchSize = (i + BATCH_SIZE > numImages) ? (numImages - i) : BATCH_SIZE;

        double host_input[BATCH_SIZE * INPUT_SIZE];
        double host_output[BATCH_SIZE * OUTPUT_SIZE];

        for (int b = 0; b < actualBatchSize; b++) {
            for (int j = 0; j < INPUT_SIZE; j++)
                host_input[b * INPUT_SIZE + j] = images[i + b][j];
        }

        CUDA_CHECK(cudaMemcpy(input_batch, host_input, actualBatchSize * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

        dim3 grid_hidden((HIDDEN_SIZE + THREADCOUNT-1) / THREADCOUNT, actualBatchSize);
        forwardHiddenKernel<<<grid_hidden, THREADCOUNT>>>(net->W1, net->b1, input_batch, hidden_batch, INPUT_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 grid_output((OUTPUT_SIZE + THREADCOUNT-1) / THREADCOUNT, actualBatchSize);
        forwardOutputKernel<<<grid_output, THREADCOUNT>>>(net->W2, net->b2, hidden_batch, output_batch, HIDDEN_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 softmax_grid(1, actualBatchSize);
        softmaxKernel<<<softmax_grid, 1>>>(output_batch, actualBatchSize, OUTPUT_SIZE);


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
    }

    double accuracy = (double)correct / numImages * 100.0;
    printf("Evaluation Accuracy: %.2f%% - Time: %.3fs\n", accuracy, get_time(eval_start));

    cudaFree(input_batch);
    cudaFree(hidden_batch);
    cudaFree(output_batch);
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
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network with CUDA\n\n");

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
