/**
 * @file k_means_openmp.c
 * @brief Algoritmo K-Means paralelo com OpenMP para dados binários do Spotify
 * =========================================================================
 * Processador de uso I9-13900 32 threads, 24 nucleos
 * TEMPOS DE EXECUÇÃO (Média obtida nos testes):
 * - Versão Sequencial (1 thread):  9.3720 segundos
 * - OpenMP 2 threads:              2.5270 segundos
 * - OpenMP 4 threads:              1.1950 segundos
 * - OpenMP 8 threads:              1.3650 segundos
 * - OpenMP 16 threads:             0.8040 segundos
 * - OpenMP 32 threads:             0.6490 segundos
 * =========================================================================
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h> 

#define DIM 12  
#define MAX_RECORDS 5000000 // Quantidade de registros para o teste

typedef struct observation
{
    double* features; 
    int group; 
} observation;

typedef struct cluster
{
    double* centroid; 
    size_t count; 
} cluster;

double calculateDistance(const observation* o, const cluster* c)
{
    double dist = 0;
    for (int i = 0; i < DIM; i++)
    {
        double diff = c->centroid[i] - o->features[i];
        dist += diff * diff;
    }
    return dist;
}

int calculateNearst(observation* o, cluster clusters[], int k)
{
    double minD = DBL_MAX;
    double dist = 0;
    int index = -1;
    for (int i = 0; i < k; i++)
    {
        dist = calculateDistance(o, &clusters[i]);
        if (dist < minD)
        {
            minD = dist;
            index = i;
        }
    }
    return index;
}

cluster* kMeans(observation observations[], size_t size, int k)
{
    cluster* clusters = malloc(sizeof(cluster) * k);
    for (int i = 0; i < k; i++) {
        clusters[i].centroid = calloc(DIM, sizeof(double));
    }

    /* Inicialização paralela dos grupos */
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < size; j++)
    {
        observations[j].group = rand() % k;
    }

    size_t changed = 0;
    size_t minAcceptedError = size / 10000; 
    int t = 0;

    do
    {
        for (int i = 0; i < k; i++)
        {
            memset(clusters[i].centroid, 0, DIM * sizeof(double));
            clusters[i].count = 0;
        }

        /* Acumulação paralela com redução local por thread */
        #pragma omp parallel
        {
            double** local_centroids = malloc(sizeof(double*) * k);
            size_t* local_counts = calloc(k, sizeof(size_t));
            for(int i = 0; i < k; i++) local_centroids[i] = calloc(DIM, sizeof(double));

            #pragma omp for schedule(static)
            for (size_t j = 0; j < size; j++)
            {
                int g = observations[j].group;
                for (int d = 0; d < DIM; d++)
                {
                    local_centroids[g][d] += observations[j].features[d];
                }
                local_counts[g]++;
            }

            #pragma omp critical
            {
                for (int i = 0; i < k; i++)
                {
                    for (int d = 0; d < DIM; d++)
                    {
                        clusters[i].centroid[d] += local_centroids[i][d];
                    }
                    clusters[i].count += local_counts[i];
                }
            }

            for(int i = 0; i < k; i++) free(local_centroids[i]);
            free(local_centroids);
            free(local_counts);
        }

        for (int i = 0; i < k; i++)
        {
            if (clusters[i].count > 0) {
                for (int d = 0; d < DIM; d++) {
                    clusters[i].centroid[d] /= clusters[i].count;
                }
            }
        }

        /*Atualização paralela dos grupos com redução */
        changed = 0; 
        #pragma omp parallel for reduction(+:changed) private(t) schedule(static)
        for (size_t j = 0; j < size; j++)
        {
            t = calculateNearst(observations + j, clusters, k);
            if (t != observations[j].group)
            {
                changed++;
                observations[j].group = t;
            }
        }

    } while (changed > minAcceptedError); 

    return clusters;
}

int main()
{
    srand(time(NULL));
    
    observation* dataset = malloc(sizeof(observation) * MAX_RECORDS);
    double* all_features = malloc(sizeof(double) * MAX_RECORDS * DIM);

    printf("Lendo dados purificados de dados.bin...\n");
    FILE* fp = fopen("dados.bin", "r");
    if (!fp) {
        perror("Erro ao abrir dados.bin. Certifique-se de gerar o arquivo primeiro");
        free(dataset); free(all_features);
        return 1;
    }

    size_t count = 0;
    while (count < MAX_RECORDS) {
        dataset[count].features = &all_features[count * DIM];
        dataset[count].group = -1;

        // Leitura direta dos 12 doubles correspondentes à linha
        int lidos = 0;
        for(int d = 0; d < DIM; d++) {
            if (fscanf(fp, "%lf", &dataset[count].features[d]) == 1) {
                lidos++;
            }
        }
        if (lidos < DIM) break; // Fim do arquivo
        count++;
    }
    fclose(fp);

    printf("%zu registros carregados com sucesso de forma segura.\n", count);

    int k = 16; 
    double start_time = omp_get_wtime();
    cluster* clusters = kMeans(dataset, count, k);
    double end_time = omp_get_wtime();
    
    printf("\nTempo de Execucao do K-Means: %.4f segundos\n", end_time - start_time);

    // Limpeza dos dados
    for (int i = 0; i < k; i++) free(clusters[i].centroid);
    free(all_features);
    free(dataset);
    free(clusters);

    return 0;
}
