//  Copyright: (c) lesya89
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <limits.h>
#include <memory.h>
#include <float.h>
#include <cstdlib>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <string>

#include "include/radix_sort.h"

#define N 1000000
#define DEBUG 0
#define MASTER_PROCESS 0

double start_time, stop_time;

template <typename ValueType>
void print_vector(ValueType *v, int vector_size) {
    int i;

    for (i = 0; i < vector_size; i++)
        printf(" %f\n ", v[i]);
}

template <typename ValueType>
ValueType *merge(ValueType *v1, int n1, ValueType *v2, int n2) {
double  *res;
    int i = 0,
        j = 0,
        index = 0;

    res = new  ValueType[n1 + n2];

    while (i < n1 && j < n2) {
        if (v1[i] < v2[j])
            res[index++] = v1[i++];
        else
            res[index++] = v2[j++];
    }

    while (i < n1)
        res[index++] = v1[i++];

    while (j < n2)
        res[index++] = v2[j++];
return res;
}

int main(int argc, char **argv) {
    int m, vector_size = N;
    double *data   = nullptr,
        *datacp = nullptr;
    double seq_time = 0,
           par_time = 0;
    int id, proc_number;
    MPI_Status status;
    int chunk_size;
    double *chunk;
    double *other;
    int step;
    int i;

    if (argc > 1) {
        vector_size = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_number);

    if (id == MASTER_PROCESS) {
        int extra_elements;
        int data_size;

        std::srand(unsigned(std::time(NULL)));

        extra_elements = vector_size % proc_number;
        chunk_size = vector_size / proc_number;
        data_size = vector_size + ((extra_elements == 0) ? 0 :
                    proc_number - extra_elements);

        /* allocate memory for vector
         * increase the size of the array so that
         * it is divided without a balance by the number of processors
         */
        data   = new double[data_size];
        datacp = new double[vector_size];

        /* fill main elements of vector
         * rand() + rand() can give overflow int,
         * thus we get negative numbers in the vector.
         * it needs to be done because
         * rand() returns non-negative numbers
         */
        for (i = 0; i < vector_size; i++)
            data[i] = static_cast<double>((std::rand() %
                      (10000 - (-10000) + 1) + (-10000)) / 100);

        /* copy data to another array for sequential sorting */
        memcpy(datacp, data, vector_size * sizeof(double));

        /* fill extra elements of vector
         * fill in the maximum number so that after sorting
         * you know which extra elements need to be thrown out
         */
        if (extra_elements != 0) {
            for (i = vector_size;
                 i < data_size;
                 i++)
                data[i] = DBL_MAX;
            chunk_size++;
        }

        if (vector_size <= 20) {
            printf("Vector before: \n");
            print_vector(data, vector_size);
        }

        start_time = MPI_Wtime();

        MPI_Bcast(&chunk_size, 1, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);
        chunk = new double[chunk_size];
        MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE,
                    MASTER_PROCESS, MPI_COMM_WORLD);

        RadixSortMSD<double>(chunk, chunk_size);
    } else {
        MPI_Bcast(&chunk_size, 1, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);
        chunk = new double[chunk_size];
        MPI_Scatter(data, chunk_size, MPI_DOUBLE, chunk, chunk_size, MPI_DOUBLE,
                    MASTER_PROCESS, MPI_COMM_WORLD);

        RadixSortMSD<double>(chunk, chunk_size);
    }

    step = 1;

    while (step < proc_number) {
        if (id % (2 * step) == 0) {
            if (id + step < proc_number) {
                MPI_Recv(&m, 1, MPI_INT, id + step, MASTER_PROCESS,
                         MPI_COMM_WORLD, &status);
                other = new double[m];
                MPI_Recv(other, m, MPI_DOUBLE, id + step, MASTER_PROCESS,
                         MPI_COMM_WORLD, &status);

                chunk = merge(chunk, chunk_size, other, m);

                delete[] other;
                chunk_size += m;
            }
        } else {
            int near = id - step;
            MPI_Send(&chunk_size, 1, MPI_INT, near, MASTER_PROCESS,
                     MPI_COMM_WORLD);
            MPI_Send(chunk, chunk_size, MPI_DOUBLE, near, MASTER_PROCESS,
                     MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (id == MASTER_PROCESS) {
        stop_time = MPI_Wtime();
        par_time = (stop_time - start_time);
        printf("\nParallel is:\n");
        printf("\n""Time: %f secs\n",                par_time);

        if (vector_size <= 20) {
            printf("Vector after: \n");
            print_vector(chunk, vector_size);
        }

        start_time = MPI_Wtime();
        RadixSortMSD<double>(datacp, vector_size);
        stop_time = MPI_Wtime();
        seq_time = (stop_time - start_time);

        printf("\nSequential is :\n");
        printf("\n"
               "Time: %f secs\n\n",
               seq_time);

        if (vector_size <= 20) {
            printf("Vector after: \n");
            print_vector(datacp, vector_size);
        }

        printf("\nVectors after sort are %s\n",
               (!memcmp(chunk, datacp, vector_size)) ? "equal" : "not equal");
        printf("Effect was: %f\n", (seq_time / par_time));

        delete[] data;
        delete[] datacp;
    }

    delete[] chunk;

    MPI_Finalize();

    return 0;
}
