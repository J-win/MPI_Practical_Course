// Copyright 2018 <Copyright Alexei Druzhinin>
#include <assert.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <cmath>

// Check result of linear and parallel
void CheckResults(int *linRes, int *parRes, int rows, int cols, int remB) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (linRes[i * cols + j] != parRes[i * cols + j + i*remB]) {
                std::cout << std::endl <<
            "Error! Linear and parallel results are not equal" << std::endl;
                return;
            }
    std::cout << std::endl << "Matrices are equal" << std::endl;
}

void FillMatrix(int *A, int rows, int cols) {  // Matrix Fill
    int count = std::rand() % 5;
    for (int i = 0; i < rows * cols; i++) {
        A[i] = count++;
    }
}

void StandartMatrixMult(int *A, int *B, int *C, int Arows, int Acols,
                    int Brows, int Bcols) {  // Linear matrix multiplication
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Bcols; j++) {
            C[i * Bcols +j] = 0;
            for (int k = 0; k < Acols; k++) {
                C[i * Bcols + j] += A[i * Acols + k] * B[k * Bcols + j];
            }
        }
    }
}

// Matrix transpose
void Transpose(int *matrixold, int *matrixnew, int rows, int cols) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            matrixnew[i * rows + j] = matrixold[j * cols + i];
        }
    }
}

void Print(int *matrix, int rows, int cols, int remA, int remB) {  // Print
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j + i*remB] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    // initialize srand and main variables
    std::srand(static_cast<int>(std::time(0)));
    int Arows = 0, Acols = 0, Brows = 0, Bcols = 0;
    MPI_Status status;
    double startTime = 0, endTime = 0, Time = 0;
    int Id = 0, numProcs = 0;
    int *MatrixA = NULL, *MatrixB = NULL, *MatrixBtr = NULL;
    int *MatrixC_l = NULL, *MatrixC = NULL;
    int* tmpA = NULL, *tmpB = NULL, *tmpC = NULL;
    int flag = 0;
    int index = 0;
    int tmp = 0;
    int partA = 0;
    int size_partA = 0;
    int partB = 0;
    int size_partB = 0;
    int size_partC = 0;
    int remainderA = 0;
    int remainderB = 0;
    int NextProc = 0;
    int PrevProc = 0;
    // variables for line block
    double startTime_l = 0;
    double endTime_l = 0;
    double Time_l = 0;
    // initialize arguments from cmd
    if (argc > 4) {
        Arows = atoi(argv[1]);
        Acols = atoi(argv[2]);
        Brows = atoi(argv[3]);
        Bcols = atoi(argv[4]);
    } else {
        Arows = 2;
        Acols = 3;
        Brows = 3;
        Bcols = 2;
    }
    // check sizes
    if (Acols != Brows) {
        std::cout << "Error in sizes" << std::endl;
        return -1;
    }

    // initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Initialized(&flag);  // check
    if (!flag) {
        std::cout << "Error in MPI_Init!!!";
        return -1;
    }

    // communicators
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &Id);

    // dividing the matrix A and B
    partA = static_cast<int>(std::ceil(static_cast<double>(Arows) /
    static_cast<double>(numProcs)));
    partB = static_cast<int>(std::ceil(static_cast<double>(Bcols) /
    static_cast<double>(numProcs)));
    size_partA = partA * Acols;
    size_partB = partB * Brows;
    size_partC = partA * partB * numProcs;
    remainderA = partA * numProcs - Arows;
    remainderB = partB * numProcs - Bcols;

    if (Id == 0) {
        // assignment the memory for matrixes
        MatrixA = new int[partA * numProcs * Acols];
        MatrixB = new int[Brows * partB * numProcs];
        MatrixC = new int[partA * numProcs * partB * numProcs];
        MatrixC_l = new int[Arows * Bcols];
        MatrixBtr = new int[Brows * partB * numProcs];

        // fill the matrixes and transpose matrix B
        FillMatrix(MatrixA, Arows, Acols);
        FillMatrix(MatrixB, Brows, Bcols);
        Transpose(MatrixB, MatrixBtr, Brows, Bcols);
        for (int i = Arows * Acols; i < partA * numProcs * Acols; i++)
            MatrixA[i] = 0;
        for (int i = Brows * Bcols; i < partB * numProcs * Brows; i++)
            MatrixBtr[i] = 0;

        // if matrixes have small size -> print
        if ((Acols < 4) && (Arows < 4) && (Bcols < 4) && (Brows < 4)) {
            std::cout << "Matrix A:" << std::endl;
            Print(MatrixA, Arows, Acols, 0, 0);
            std::cout << "Matrix B:" << std::endl;
            Print(MatrixB, Brows, Bcols, 0, 0);
            std::cout << "Matrix Btr:" << std::endl;
        }

// LINE BLOCK
        startTime_l = MPI_Wtime();
        StandartMatrixMult(MatrixA, MatrixB, MatrixC_l, Arows,
                              Acols, Brows, Bcols);
        // Line Results
        endTime_l = MPI_Wtime();
        std::cout << std::endl << "===LINE===" << std::endl;
        if ((Arows < 5) && (Bcols < 5)) {
            std::cout << "Matrix C:" << std::endl;
            Print(MatrixC_l, Arows, Bcols, 0, 0);
        }
        Time_l = endTime_l - startTime_l;
        std::cout << "Time: " << Time_l << " sec" << std::endl;
// END LINE BLOCK

// PARALLEL BLOCK
        startTime = MPI_Wtime();
    }
    tmpA = new int[size_partA];
    tmpB = new int[size_partB];
    tmpC = new int[size_partC];
    for (int i = 0; i < size_partA; i++)
        tmpA[i] = 0;
    for (int i = 0; i < size_partB; i++)
        tmpB[i] = 0;
    for (int i = 0; i < size_partC; i++)
        tmpC[i] = 0;

    MPI_Scatter(MatrixA, size_partA, MPI_INT, tmpA, size_partA,
                MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(MatrixBtr, size_partB, MPI_INT, tmpB, size_partB,
             MPI_INT, 0, MPI_COMM_WORLD);

    // calculation of elements on the main diagonal
    for (int i = 0; i < partA; i++) {
        for (int j = 0; j < partB; j++) {
            tmpC[i * partB * numProcs + j + partB * Id] = 0;
            for (int k = 0; k < Acols; k++) {
                tmpC[i * partB * numProcs + j + partB * Id] +=
                        tmpA[i * Acols + k] * tmpB[j * Brows + k];
            }
        }
    }
    NextProc = Id + 1;
    if (Id == numProcs - 1)
        NextProc = 0;
    PrevProc = Id - 1;
    if (Id == 0)
        PrevProc = numProcs - 1;

    // cyclic exchange of columns matrix B
    for (int nm = 1; nm < numProcs; nm++) {
        MPI_Sendrecv_replace(tmpB, size_partB, MPI_INT, NextProc, 0,
            PrevProc, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < partA; i++) {
            for (int j = 0; j < partB; j++) {
                tmp = 0;
                for (int k = 0; k < Acols; k++)
                    tmp += tmpA[i * Acols + k] * tmpB[j * Brows + k];
                if (Id - nm >= 0)
                    index = Id - nm;
                else
                    index = (Id - nm + numProcs);
                tmpC[i * partB * numProcs + j + index * partB] = tmp;
            }
        }
    }
    // assembly of the resulting matrix
    MPI_Gather(tmpC, size_partC, MPI_INT, MatrixC, size_partC,
    MPI_INT, 0, MPI_COMM_WORLD);
    // Parallel results
    if (Id == 0) {
        endTime = MPI_Wtime();
        std::cout << std::endl <<"===PARALLEL===" << std::endl;
        if ((Arows < 5) && (Bcols < 5)) {
            std::cout << "Matrix C:" << std::endl;
            Print(MatrixC, Arows, Bcols, remainderA, remainderB);
        }
        Time = endTime - startTime;
        std::cout << "Time: " << Time << " sec" << std::endl;
        std::cout << "==============";
        CheckResults(MatrixC_l, MatrixC, Arows, Bcols, remainderB);
    }
    MPI_Finalize();
    if (tmpA != NULL ) delete []tmpA;
    if (tmpB != NULL ) delete []tmpB;
    if (tmpC != NULL ) delete []tmpC;
// END PARALLEL BLOCK
    if (MatrixA != NULL ) delete []MatrixA;
    if (MatrixB != NULL ) delete []MatrixB;
    if (MatrixC_l != NULL ) delete []MatrixC_l;
    if (MatrixC != NULL ) delete []MatrixC;
    if (MatrixBtr != NULL ) delete []MatrixBtr;
    return 0;
}
