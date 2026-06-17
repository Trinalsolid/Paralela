/**
 * @file k_means_cuda.cu
 * @brief Algoritmo K-Means com aceleração de GPU nativa utilizando CUDA
 * =========================================================================
 * Placa de Vídeo de Uso: RTX 4060TI 16GB
 * =========================================================================
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define DIM 12  
#define MAX_RECORDS 5000000 

// Estruturas de dados idênticas às anteriores, ideais para cópia direta via cudaMemcpy
typedef struct observation
{
    double features[DIM]; 
    int group; 
} observation;

typedef struct cluster
{
    double centroid[DIM]; 
    size_t count; 
} cluster;

// Macro para tratamento e captura de erros de API do CUDA
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s:%d na funcao %s -> %s\n", file, line, func, cudaGetErrorString(err));
        exit(1);
    }
}

// __device__ indica que a função roda exclusivamente dentro dos núcleos da GPU
__device__ double calculateDistanceCUDA(const observation* o, const cluster* c)
{
    double dist = 0;
    for (int i = 0; i < DIM; i++)
    {
        double diff = c->centroid[i] - o->features[i];
        dist += diff * diff;
    }
    return dist;
}

__device__ int calculateNearstCUDA(const observation* o, const cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    for (int i = 0; i < k; i++)
    {
        dist = calculateDistanceCUDA(o, &clusters[i]);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

// Inicialização paralela dos grupos baseada no índice global da thread
__global__ void kernelInitGroups(observation* observations, size_t size, int k)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        observations[idx].group = (idx * 11) % k;
    }
}

// Reseta os acumuladores dos centroides na GPU
__global__ void kernelResetClusters(cluster* clusters, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k)
    {
        for (int d = 0; d < DIM; d++) {
            clusters[idx].centroid[d] = 0.0;
        }
        clusters[idx].count = 0;
    }
}

// Acumulação paralela dos dados usando atomicAdd nativo para double em hardware
__global__ void kernelAccumulateClusters(const observation* observations, size_t size, cluster* clusters)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int g = observations[idx].group;
        for (int d = 0; d < DIM; d++)
        {
            // atomicAdd para double é suportado nativamente a partir de arquiteturas Pascal+ (RTX 4060Ti inclusa)
            atomicAdd(&(clusters[g].centroid[d]), observations[idx].features[d]);
        }
        // Como count é size_t (unsigned long long), usamos casting para casar com a assinatura atômica do CUDA
        atomicAdd((unsigned long long*)&(clusters[g].count), 1ULL);
    }
}

// Calcula a média aritmética final de cada dimensão dos clusters
__global__ void kernelComputeAverages(cluster* clusters, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k)
    {
        unsigned long long current_count = clusters[idx].count;
        if (current_count > 0)
        {
            for (int d = 0; d < DIM; d++)
            {
                clusters[idx].centroid[d] /= current_count;
            }
        }
    }
}

// Atualiza os clusters mais próximos e faz a contagem de mudanças por bloco (redução parcial)
__global__ void kernelUpdateGroups(observation* observations, size_t size, const cluster* clusters, int k, size_t* d_changed)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_changed = 0;

    if (idx < size)
    {
        int t = calculateNearstCUDA(&observations[idx], clusters, k);
        if (t != observations[idx].group)
        {
            local_changed = 1;
            observations[idx].group = t;
        }
    }

    // Redução atômica rápida para somar todas as mudanças da grade na variável global
    if (local_changed > 0)
    {
        atomicAdd((unsigned long long*)d_changed, 1ULL);
    }
}

void kMeansCUDA(observation h_observations[], size_t size, cluster h_clusters[], int k)
{
    observation* d_observations = NULL;
    cluster* d_clusters = NULL;
    size_t* d_changed = NULL;
    size_t h_changed = 0;
    size_t minAcceptedError = size / 10000;

    // Configuração de Threads e Blocos para cobrir 5 milhões de registros
    int threadsPerBlock = 256;
    int blocksPerGridRecords = (size + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridClusters = (k + threadsPerBlock - 1) / threadsPerBlock;

    // Alocação de memória física na VRAM da GPU (Device)
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_observations, sizeof(observation) * size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_clusters, sizeof(cluster) * k));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_changed, sizeof(size_t)));

    // Cópia inicial do Dataset da CPU para a GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_observations, h_observations, sizeof(observation) * size, cudaMemcpyHostToDevice));

    // Inicializa os grupos direto na GPU
    kernelInitGroups<<<blocksPerGridRecords, threadsPerBlock>>>(d_observations, size, k);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    do
    {
        // 1. Reseta os acumuladores na GPU
        kernelResetClusters<<<blocksPerGridClusters, threadsPerBlock>>>(d_clusters, k);

        // 2. Acumula os valores das dimensões na GPU de forma massiva
        kernelAccumulateClusters<<<blocksPerGridRecords, threadsPerBlock>>>(d_observations, size, d_clusters);

        // 3. Divide as somas pela quantidade para obter o novo centroide
        kernelComputeAverages<<<blocksPerGridClusters, threadsPerBlock>>>(d_clusters, k);

        // Reset do contador de controle de convergência na GPU
        h_changed = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_changed, &h_changed, sizeof(size_t), cudaMemcpyHostToDevice));

        // 4. Reposiciona as observações nos novos grupos e computa o erro
        kernelUpdateGroups<<<blocksPerGridRecords, threadsPerBlock>>>(d_observations, size, d_clusters, k, d_changed);
        
        // Sincroniza e puxa o contador de mudanças de volta para a CPU avaliar a condição de parada
        CHECK_CUDA_ERROR(cudaMemcpy(&h_changed, d_changed, sizeof(size_t), cudaMemcpyDeviceToHost));

    } while (h_changed > minAcceptedError);

    // Copia os resultados de volta para a RAM da CPU
    CHECK_CUDA_ERROR(cudaMemcpy(h_observations, d_observations, sizeof(observation) * size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_clusters, d_clusters, sizeof(cluster) * k, cudaMemcpyDeviceToHost));

    // Limpeza obrigatória de memória na GPU
    cudaFree(d_observations);
    cudaFree(d_clusters);
    cudaFree(d_changed);
}

int main()
{
    observation* dataset = (observation*)malloc(sizeof(observation) * MAX_RECORDS);
    if (!dataset) {
        printf("Erro ao alocar memoria para o dataset na CPU.\n");
        return 1;
    }

    printf("Lendo dados purificados de dados.bin...\n");
    FILE* fp = fopen("dados.bin", "r");
    if (!fp) {
        perror("Erro ao abrir dados.bin");
        free(dataset);
        return 1;
    }

    size_t count = 0;
    while (count < MAX_RECORDS) {
        dataset[count].group = -1;
        int lidos = 0;
        for(int d = 0; d < DIM; d++) {
            if (fscanf(fp, "%lf", &dataset[count].features[d]) == 1) {
                lidos++;
            }
        }
        if (lidos < DIM) break;
        count++;
    }
    fclose(fp);

    printf("%zu registros carregados com sucesso.\n", count);

    int k = 16; 
    cluster* clusters = (cluster*)calloc(k, sizeof(cluster));

    printf("Iniciando K-Means na GPU nativa via CUDA...\n");
    
    // Medição de tempo nativa usando eventos CUDA
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kMeansCUDA(dataset, count, clusters, k);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("\nTempo de Execucao do K-Means em CUDA: %.4f segundos\n", milliseconds / 1000.0);

    // Destruição dos eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(dataset);
    free(clusters);
    return 0;
}
