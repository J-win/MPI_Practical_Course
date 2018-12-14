// copyright : (C) by Pahandrovich
#include <mpi.h>
#include <assert.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cmath>

int check(double *a, double *b, int size) {
  int i = 0;
  while (i < size && a[i] == b[i]) i++;
  if (i == size)
    return 1;
  else
    return 0;
}

int main(int argc, char* argv[]) {
  std::srand(static_cast<int>(time(NULL)));
  int rank, CountP, flag;
  if (argc < 3) {
    std::cout << "Error";
    return -1;
  }
  int rows = atoi(argv[1]);
  int cols = atoi(argv[2]);
  if (rows <= 0 || cols <= 0) {
    std::cout << "Error";
    return -1;
  }

  MPI_Init(&argc, &argv);
  MPI_Initialized(&flag);
  if (!flag) {
    std::cout << "Error";
    return -1;
  }
  MPI_Comm_size(MPI_COMM_WORLD, &CountP);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double *vect = NULL,
    *matrix = NULL;
  double Time_begin = 0;
  double Time_end = 0;
  int Matrix_size = rows * cols;
  int Vect_size = cols;
  int Result_size = rows;
  double *Result = NULL;
  double *Result_l = NULL;
  double *tempRes = NULL;
  double *subMatr = NULL;
  double *subVect = NULL;

  int submit_num = static_cast<int>(ceil(cols / CountP));
  tempRes = new double[Result_size];
  subMatr = new double[submit_num*rows];
  subVect = new double[submit_num];

  for (int i = 0; i < submit_num*rows; i++) {
    if (i < submit_num) subVect[i] = 0.0;
    subMatr[i] = 0.0;
  }
  for (int i = 0; i < Result_size; i++) tempRes[i] = 0.0;

  if (rank == 0) {
    vect = new double[Vect_size];
    matrix = new double[Matrix_size];
    Result = new double[Result_size];
    Result_l = new double[Result_size];
    //   -----------------Initialization-----------------
    for (int i = 0; i < Matrix_size; i++) {
      if (i < Vect_size) vect[i] = std::rand() % 10 - 5;
      if (i < Result_size) Result[i] = 0.0;
      if (i < Result_size) Result_l[i] = 0.0;
      matrix[i] = std::rand() % 10 - 5;
    }
    //   -----------Print-----------------
    if (rows < 11 && cols < 11) {
      std::cout << "Vector:" << std::endl;
      for (int i = 0; i < Vect_size; i++)
        std::cout << " " << vect[i];
      std::cout << "\nMatrix:" << std::endl;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
          std::cout << " " << matrix[i + rows * j];
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "Matr size = " << Matrix_size << std::endl;
    std::cout << "Vector size = " << Vect_size << std::endl;
    std::cout << "res size = " << Result_size << std::endl;
    std::cout << "rows = " << rows << std::endl;
    std::cout << "cols = " << cols << std::endl;
    std::cout << "rows * cols = " << rows * cols << std::endl << std::endl;
    //  --------------  Line  --------------------
    Time_begin = MPI_Wtime();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++)
        Result_l[i] += matrix[i + rows * j] * vect[j];
        }
    Time_end = MPI_Wtime();
    std::cout << "Line_Time = " << Time_end - Time_begin << std::endl;
    if (rows < 11 && cols < 11) {
      std::cout << "Line_Result: " << std::endl;
      for (int i = 0; i < Result_size; i++)
      std::cout << Result_l[i] << std::endl;
    }
    std::cout << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  Time_begin = MPI_Wtime();

  //  Send
  MPI_Scatter(vect, submit_num, MPI_DOUBLE,
        subVect, submit_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(matrix, submit_num*rows, MPI_DOUBLE,
        subMatr, submit_num*rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //  Work
  for (int k = 0; k < submit_num; k++)
    for (int j = 0; j < rows; j++)
      tempRes[j] += subVect[k] * subMatr[j + rows * k];

  //  Sum result
  MPI_Reduce(tempRes, Result, Result_size, MPI_DOUBLE,
      MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  Time_end = MPI_Wtime();

  if (rank == 0) {
    std::cout << std::endl << "Parallel time: ";
    std::cout << Time_end - Time_begin << std::endl;
    std::cout << "check = " <<
        check(Result, Result_l, Result_size) << std::endl;
    if (rows < 11 && cols < 11) {
    std::cout << "Parallel_Result: " << std::endl;
    for (int i = 0; i < Result_size; i++)
      std::cout << Result[i] << std::endl;
    }
    std::cout << std::endl;
  }

  if (vect != NULL) delete[]vect;
  if (matrix != NULL) delete[]matrix;
  if (subMatr != NULL) delete[]subMatr;
  if (subVect != NULL) delete[]subVect;
  if (Result != NULL) delete[]Result;
  if (Result_l != NULL) delete[]Result_l;
  if (tempRes != NULL) delete[]tempRes;

  MPI_Finalize();

  return 0;
}
