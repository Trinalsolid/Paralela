# Paralela
Trabalho de Computação Paralela com OpenMP, OpenMP para GPU, e CUDA para GPU

## Estrutura
```
├── Dataset/              # Pasta de arquivo
│   └── Arquivos do dataset e o.zip     
├── OpemMP/               # Pasta de arquivo
│   ├── dados.bin            # Arquivo binário do dataset
│   ├── k_means_openmp.c     # Código openmp
│   └── k-means.txt          # Explicação de como executar o OpemMP seq e paralelo
├── OpenMP para GPU/     # Pasta de arquivo
│   ├── dados.bin            # Arquivo binário do dataset
│   ├── k_means_openmp.c     # Código openmp para GPU
│   └── k-means.txt          # Explicação de como executar o OpemMP para GPU ou emular a GPU na CPU
└── CUDA para GPU/     # Pasta de arquivo
    ├── dados.bin            # Arquivo binário do dataset
    ├── k_means_cuda.cu       # Código CUDA para GPU
    └── k-means.txt          # Explicação de como executar o OpemMP para GPU ou emular a GPU na CPU
```
## Dataset utilizado
```
https://www.kaggle.com/datasets/serkantysz/490k-spotify-song-audio-embeddings-and-metadata
```
### Executar o Código para extair os binarios do dataset
```bash
$ head -n 5000000 ../Dataset/songs.csv | awk -F, '{print $3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14}' | tr ' ' '\n' | awk '{printf "%f ", $1}' > dados.bin
```
1. opção A  Coloque o dados.bin em cada uma das pastas
2. opção B  Coloque o dados.bin em um pasta e altere o caminho dos código para a pasta desejada

## 👥 Integrantes do Grupo

  - [Juan Pablo Ramos de Oliveira]
  - [Luiz Gabriel Milione Assis]
  - [Luís Fernando Rdorigues Braga]
