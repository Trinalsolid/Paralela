/**
 * @file k_means_gpu.c
 * @brief Algoritmo K-Means com aceleração de GPU utilizando OpenMP Target Offloading
 * =========================================================================
 * Placa de Vídeo de Uso: RTX 4060TI 16GB
 * TEMPOS DE EXECUÇÃO (Média obtida nos testes):
 * - Versão Sequencial (1 thread CPU):  9.3720 segundos (Ref)
 * - OpenMP GPU Offloading:            20.9390 segundos
 * =========================================================================
 */
// ===================================================
// Rodando na MX110 2 Gb
// Lendo dados purificados de dados.bin...
// 5000000 registros carregados com sucesso.
// Iniciando K-Means na GPU via OpenMP Target Offloading...

// Tempo de Execucao do K-Means na GPU: 12.9714 segundos
// free(): invalid size
// Abortado (imagem do núcleo gravada)

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h> 

#define DIM 12  
#define MAX_RECORDS 5000000 

typedef struct observation
{
    double features[DIM]; // Convertido para array fixo para mapeamento direto na GPU
    int group; 
} observation;

typedef struct cluster
{
    double centroid[DIM]; // Convertido para array fixo
    size_t count; 
} cluster;

// Funções inline declaradas para que possam ser executadas dentro dos núcleos da GPU
#pragma omp declare target
double calculateDistanceGPU(const observation* o, const cluster* c)
{
    double dist = 0;
    for (int i = 0; i < DIM; i++)
    {
        double diff = c->centroid[i] - o->features[i];
        dist += diff * diff;
    }
    return dist;
}

int calculateNearstGPU(const observation* o, const cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    for (int i = 0; i < k; i++)
    {
        dist = calculateDistanceGPU(o, &clusters[i]);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}
#pragma omp end declare target

void kMeansGPU(observation observations[], size_t size, cluster clusters[], int k)
{
    /* Cria o bloco de dados na GPU e mapeia os arrays de observações e clusters */
    #pragma omp target data map(tofrom: observations[0:size]) map(alloc: clusters[0:k])
    {
        /* Inicialização paralela dos grupos diretamente nos núcleos da GPU */
        #pragma omp target teams distribute parallel for schedule(static)
        for (size_t j = 0; j < size; j++)
        {
            //  uma operação matemática baseada no índice para gerar grupos iniciais pseudo-aleatórios na GPU.
            observations[j].group = (j * 11) % k;
        }

        size_t changed = 0;
        size_t minAcceptedError = size / 10000; 

        do
        {
            // Reseta clusters de forma paralela na GPU
            #pragma omp target teams distribute parallel for schedule(static)
            for (int i = 0; i < k; i++)
            {
                for(int d = 0; d < DIM; d++) clusters[i].centroid[d] = 0;
                clusters[i].count = 0;
            }

            /*Redução paralela otimizada na GPU.
            Como alocações de ponteiros duplos locais (malloc) não operam eficientemente na GPU,
            utilizamos uma redução combinada de matrizes gerenciada pelo próprio Target do OpenMP.
            #pragma omp target teams distribute parallel for schedule(static)*/
            
            for (size_t j = 0; j < size; j++)
            {
                int g = observations[j].group;
                // Operações atômicas na GPU garantem consistência ao consolidar dados de linhas diferentes
                for (int d = 0; d < DIM; d++)
                {
                    #pragma omp atomic
                    clusters[g].centroid[d] += observations[j].features[d];
                }
                #pragma omp atomic
                clusters[g].count++;
            }

            // Calcula as médias das coordenadas dos centroides na GPU
            #pragma omp target teams distribute parallel for schedule(static)
            for (int i = 0; i < k; i++)
            {
                if (clusters[i].count > 0) {
                    for (int d = 0; d < DIM; d++) {
                        clusters[i].centroid[d] /= clusters[i].count;
                    }
                }
            }

            /* Atualização paralela dos grupos com redução na GPU usando a cláusula reduction(+:changed) */
            changed = 0; 
            #pragma omp target teams distribute parallel for reduction(+:changed) schedule(static)
            for (size_t j = 0; j < size; j++)
            {
                int t = calculateNearstGPU(&observations[j], clusters, k);
                if (t != observations[j].group)
                {
                    changed++;
                    observations[j].group = t;
                }
            }

        } while (changed > minAcceptedError); 
    }
}

int main()
{
    // Alocação contígua na CPU usando arrays 
    observation* dataset = malloc(sizeof(observation) * MAX_RECORDS);
    if (!dataset) {
        printf("Erro ao alocar memoria para o dataset.\n");
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
    count = fread(dataset, sizeof(observation), MAX_RECORDS, fp);
    fclose(fp);

    printf("%zu registros carregados com sucesso.\n", count);

    int k = 16; 
    cluster* clusters = calloc(k, sizeof(cluster));

    printf("Iniciando K-Means na GPU via OpenMP Target Offloading...\n");
    
    double start_time = omp_get_wtime();
    kMeansGPU(dataset, count, clusters, k);
    double end_time = omp_get_wtime();
    
    printf("\nTempo de Execucao do K-Means na GPU: %.4f segundos\n", end_time - start_time);

    free(dataset);
    free(clusters);
    return 0;
}
