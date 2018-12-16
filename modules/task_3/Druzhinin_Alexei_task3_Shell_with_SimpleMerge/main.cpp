// copyright : (C) by Druzhinin Alexei
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

void Print(const double *array, int size) {
    std::cout << "Array: ";
    for (int i = 0; i < size; i++) {
        if (i == size - 1)
            std::cout << array[i];
        else
            std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

void Check(const double *first_array, const double *second_array, int size) {
    for (int i = 0; i < size; i++) {
        if (first_array[i] != second_array[i]) {
            std::cout << std::endl <<
            "Error! Linear and parallel results are not equal" << std::endl;
            return;
        }
    }
    std::cout << std::endl << "Results are equal" << std::endl;
}

void SortShell(double *array, int size) {
    int i, j, k;
    double t;
    for (k = size / 2; k > 0; k /= 2)
        for (i = k; i < size; i++) {
            t = array[i];
        for (j = i; j >= k; j -= k) {
            if (t < array[j - k])
                array[j] = array[j - k];
            else
                break;
        }
    array[j] = t;
    }
}

double *SortMerge(double *first_array, double *second_array, int first_size,
                int second_size) {
    double *array = new double[first_size+second_size];
    int i = 0, j = 0, k = 0;
    double tmp1 = 0, tmp2 = 0;
    while (i < first_size && j < second_size) {
        tmp1 = first_array[i];
        tmp2 = second_array[j];
        if (tmp1 <= tmp2) {
            array[k] = tmp1;
            i++;
        } else {
            array[k] = tmp2;
            j++;
        }
        k++;
    }
    while (i < first_size)
        array[k++] = first_array[i++];
    while (j < second_size)
        array[k++] = second_array[j++];
    return array;
}

int main(int argc, char* argv[]) {
    std::srand(static_cast<int>(time(NULL)));
    int id, numProcs, flag;
    int size;
    int remainder;
    MPI_Status status;
    double *rem_array = NULL;
    double *array = NULL;
    double *result_array = NULL;
    double *array_l = NULL;
    double *tmp_array = NULL;
    double *second_array = NULL;
    double start_time = 0;
    double end_time = 0;
    double start_time_l = 0;
    double end_time_l = 0;
    int part_size = 0;
    if (argc < 2) {
        std::cout << "Error in cmd's argument";
        return -1;
    }
    size = atoi(argv[1]);
    // MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Initialized(&flag);
    if (!flag) {
        std::cout << "Error in MPI Init";
        return -1;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (numProcs > size) {
        std::cout << "Error! Count of procs > size of array";
        return -1;
    }
    part_size = size / numProcs;
    remainder = size % numProcs;
    if (id == 0) {
        array = new double[size];
        result_array = new double[size];
        array_l = new double[size];
        for (int i = 0; i < size; i++)
            array[i] = array_l[i] = std::rand()%100000;
        if (size <= 20) {
            Print(array, size);
        }
        std::cout << std::endl;
        // LINE BLOCK
        start_time_l = MPI_Wtime();
        SortShell(array_l, size);
        end_time_l = MPI_Wtime();
        // LINE RESULT
        std::cout << "Line time: " << end_time_l - start_time_l << std::endl;
        if (size <= 20) {
            Print(array_l, size);
        }
        std::cout << std::endl;
    }
    // PARALLEL BLOCK
    tmp_array = new double[part_size];
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    // SEND PARTS OF ARRAY
    MPI_Scatter(array, part_size, MPI_DOUBLE,
    tmp_array, part_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // SORT SHELL
    SortShell(tmp_array, part_size);
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        second_array = new double[part_size];
        for (int i = 0; i < part_size; i++)
            result_array[i] = tmp_array[i];
        for (int i = 1; i < numProcs; i++) {
            // SORT MERGE
            MPI_Recv(second_array, part_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
            &status);
            result_array = SortMerge(result_array, second_array, i*part_size,
            part_size);
        }
    } else {
        MPI_Send(tmp_array, part_size, MPI_DOUBLE, 0 , 0 , MPI_COMM_WORLD);
    }
    if (id == 0) {
        if (remainder) {
            rem_array = new double[remainder];
            for (int i = 0; i < remainder; i++)
                rem_array[i] = array[size - remainder + i];
            SortShell(rem_array, remainder);
            result_array = SortMerge(result_array, rem_array, size - remainder,
            remainder);
        }
        end_time = MPI_Wtime();
    // PARALLEL RESULT
        std::cout << std::endl << "Parallel time: ";
        std::cout << end_time - start_time << std::endl;
        if (size <= 20) {
            Print(result_array, size);
        }
        std::cout << std::endl;
        Check(result_array, array_l, size);
    }
    if (array != NULL) delete[]array;
    if (array_l != NULL) delete[]array_l;
    if (tmp_array != NULL) delete[]tmp_array;
    if (result_array != NULL) delete[]result_array;
    if (rem_array != NULL) delete[]rem_array;
    if (second_array != NULL) delete[]second_array;
    MPI_Finalize();
    return 0;
}
