// copyright            : (C) 2018 by Yury
#include <assert.h>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <utility>

#define MAX_CHECKING_SET_POWER 100

double given3dFunction(double x1, double x2, double x3) {
  double res1 = std::exp(-x1 * x2);
  double res2 = std::sin(x2 + x3) / (x2 + x3);
  double res3 = std::cos(x1 * x3);
  return res1 + res2 + res3;
}
double given1dFunction(double a1) {
  return a1*a1;
}

double riman3dIntegral(int setPower, double boundaries[3][2]) {
  double a1 = boundaries[0][0];
  double b1 = boundaries[0][1];
  double a2 = boundaries[1][0];
  double b2 = boundaries[1][1];
  double a3 = boundaries[2][0];
  double b3 = boundaries[2][1];
  if ((a1 >= b1) || (a2 >= b2) || (a3 >= b3)) {
    return 0;
  }
  double sum = 0;
  for (int i = 0; i < setPower; i++) {
    double innerSum1 = 0;
    for (int j = 0; j < setPower; j++) {
      double innerSum2 = 0;
      for (int k = 0; k < setPower; k++) {
        double xI = a1 + (b1 - a1) * (static_cast<double>(i) / setPower);
        double yI = a2 + (b2 - a2) * (static_cast<double>(j) / setPower);
        double zI = a3 + (b3 - a3) * (static_cast<double>(k) / setPower);
        innerSum2 += given3dFunction(xI, yI, zI);
      }
      innerSum1 += innerSum2 / setPower;
    }
    sum += innerSum1 / setPower;
  }
  sum /= setPower;
  return sum * (b1 - a1) * (b2 - a2) * (b3 - a3);
}

double myRand(double fMin, double fMax) {
    int iRand = std::rand();
    double f = (static_cast<double>(iRand)) / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sequence3dMonteCarloSum(int setPower, double boundaries[3][2]) {
  int seed = static_cast<int>(MPI_Wtime());
  std::srand(seed);
  double part = 0;
  double a1 = boundaries[0][0];
  double b1 = boundaries[0][1];
  double a2 = boundaries[1][0];
  double b2 = boundaries[1][1];
  double a3 = boundaries[2][0];
  double b3 = boundaries[2][1];
  if ((a1 >= b1) || (a2 >= b2) || (a3 >= b3)) {
    return 0;
  }
  for (int i = 0; i < setPower; i++) {
    double innerPart1 = 0;
    for (int j = 0; j < setPower; j++) {
      double innerPart2 = 0;
      for (int k = 0; k < setPower; k++) {
        double randomArg1 = myRand(a1, b1);
        double randomArg2 = myRand(a2, b2);
        double randomArg3 = myRand(a3, b3);
        double val = given3dFunction(randomArg1, randomArg2, randomArg3);
        innerPart2 += val;
      }
      innerPart1 += innerPart2 / setPower;
    }
    part += innerPart1 / setPower;
  }
  return part * (b1 - a1) * (b2 - a2) * (b3 - a3) / setPower;
}

double partial3dMonteCarloSum(
  int setPower, int procNum,  int rank, double boundaries[3][2]
  ) {
  double a1 = boundaries[0][0];
  double b1 = boundaries[0][1];
  double a2 = boundaries[1][0];
  double b2 = boundaries[1][1];
  double a3 = boundaries[2][0];
  double b3 = boundaries[2][1];
  if ((a1 >= b1) || (a2 >= b2) || (a3 >= b3)) {
    return 0;
  }
  // initialize specific seed
  int seed = static_cast<int>(MPI_Wtime()) * rank;
  std::srand(seed);
  double partialSum = 0;
  for (int i = 0; i < setPower; i++) {
    double innerPart1 = 0;
    for (int j = 0; j < setPower; j++) {
      double innerPart2 = 0;
      for (int k = 0; k < setPower / procNum; k++) {
        double randomArg1 = myRand(a1, b1);
        double randomArg2 = myRand(a2, b2);
        double randomArg3 = myRand(a3, b3);
        double val = given3dFunction(randomArg1, randomArg2, randomArg3);
        innerPart2 += val;
      }
      innerPart1 += innerPart2 / setPower;
    }
    partialSum += innerPart1 / setPower;
  }
  return partialSum * (b1 - a1) * (b2 - a2) * (b3 - a3) / setPower;
}

int main(int argc, char * argv[]) {
  int status = 0, rank = 0, procNum = 0;
  int setPower = atoi(argv[1]);
  double localMonteCarloIntegral = 0;
  double globalMonteCarloIntegral = 0;
  double parTime1 = 0, parTime2 = 0,
    seqTime1 = 0, seqTime2 = 0,
    rimanTime1 = 0, rimanTime2 = 0;
  double boundaries[3][2] = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  for (int i = 0; i < 3; i ++) {
    boundaries[i][0] = atof(argv[2 + 2 * i]);
    boundaries[i][1] = atof(argv[3 + 2 * i]);
  }
  status = MPI_Init(& argc, & argv);
  // assert(status == MPI_SUCCESS);
  if (status != MPI_SUCCESS) {
    std::cout << "ERROR\n";
    return 1;
  }

  status = MPI_Comm_rank(MPI_COMM_WORLD, & rank);
  // assert(status == MPI_SUCCESS);
  if (status != MPI_SUCCESS) {
    std::cout << "ERROR\n";
    return 1;
  }

  status = MPI_Comm_size(MPI_COMM_WORLD, & procNum);
  // assert(status == MPI_SUCCESS);
  if (status != MPI_SUCCESS) {
    std::cout << "ERROR\n";
    return 1;
  }
  // Parallel version
  // чтобы были разные наборы
  MPI_Barrier(MPI_COMM_WORLD);
  parTime1 = MPI_Wtime();
  localMonteCarloIntegral =
    partial3dMonteCarloSum(setPower, procNum, rank, boundaries);
  MPI_Reduce(&localMonteCarloIntegral, &globalMonteCarloIntegral, 1,
    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  parTime2 = MPI_Wtime();
  if (rank == 0) {
    std::cout << "Algorhytm\t\tResult\t\t\tTime consumed\n\n";
    std::cout << "Parallel\t\t" <<
      globalMonteCarloIntegral << "\t\t" <<
      parTime2 - parTime1 << " sec\n";
    seqTime1 = MPI_Wtime();
    std::cout << "Sequence\t\t" <<
      sequence3dMonteCarloSum(setPower, boundaries) << "\t\t";
    seqTime2 = MPI_Wtime();
    std::cout << seqTime2 - seqTime1 << " sec\n";
    if (setPower < MAX_CHECKING_SET_POWER) {
      rimanTime1 = MPI_Wtime();
        std::cout << "Riman\t\t\t" <<
      riman3dIntegral(setPower, boundaries) << "\t\t";
        rimanTime2 = MPI_Wtime();
      std::cout << rimanTime2 - rimanTime1 << " sec\n";
    }
  }
  status = MPI_Finalize();
  assert(status == MPI_SUCCESS);
  return 0;
}
