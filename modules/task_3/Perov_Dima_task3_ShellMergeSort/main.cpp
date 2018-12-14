// copyright : (C) by diper1998
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

void GetVectorInfo(const double *myVector, size_t mySize) {
  std::cout << std::endl << "Size: " << mySize << std::endl;
  std::cout << "Vector: ";
  for (unsigned i = 0; i < mySize; i++) {
    std::cout << myVector[i] << ", ";
  }
  std::cout << std::endl;
}

int Test(const double *firstVector, const double *secondVector, size_t mySize) {
  for (unsigned i = 0; i < mySize; i++) {
    if (firstVector[i] != secondVector[i]) {
      return 0;
    }
  }
  return 1;
}

void SortShell(double *myVector, size_t mySize) {
  unsigned i, j, k;
  double t;
  for (k = mySize / 2; k > 0; k /= 2)
    for (i = k; i < mySize; i++) {
      t = myVector[i];
      for (j = i; j >= k; j -= k) {
        if (t < myVector[j - k])
          myVector[j] = myVector[j - k];
        else
          break;
      }
      myVector[j] = t;
    }
}

double *SortMerge(double *myVector, double *myBuffer, unsigned borderLeft,
                  unsigned borderRight) {
  if (borderLeft == borderRight) {
    myBuffer[borderLeft] = myVector[borderLeft];
    return myBuffer;
  }

  unsigned borderMiddle = (borderLeft + borderRight) / 2;

  // divide and rule
  double *bufferLeft = SortMerge(myVector, myBuffer, borderLeft, borderMiddle);
  double *bufferRight =
      SortMerge(myVector, myBuffer, borderMiddle + 1, borderRight);

  // merge of two sorted halves
  double *target = bufferLeft == myVector ? myBuffer : myVector;

  unsigned currentLeft = borderLeft, currentRight = borderMiddle + 1;
  for (unsigned i = borderLeft; i <= borderRight; i++) {
    if (currentLeft <= borderMiddle && currentRight <= borderRight) {
      if (bufferLeft[currentLeft] < bufferRight[currentRight]) {
        target[i] = bufferLeft[currentLeft];
        currentLeft++;
      } else {
        target[i] = bufferRight[currentRight];
        currentRight++;
      }
    } else if (currentLeft <= borderMiddle) {
      target[i] = bufferLeft[currentLeft];
      currentLeft++;
    } else {
      target[i] = bufferRight[currentRight];
      currentRight++;
    }
  }
  return target;
}

int main(int argc, char *argv[]) {
  srand(static_cast<int>(time(0)));

  double *plVector;
  double *lnVector;
  size_t vectorSize;

  std::string commandArgument;
  commandArgument = argv[1];
  vectorSize = atoi(commandArgument.c_str());

  plVector = new double[vectorSize];
  lnVector = new double[vectorSize];

  // for parallel block

  double plStartTime = 0;
  double plEndTime = 0;
  double plWorkTime = 0;

  int flag;
  int myId, numProcs;

  // for line block

  double lnStartTime = 0;
  double lnEndTime = 0;
  double lnWorkTime = 0;

  // Initialize the MPI environment

  MPI_Init(&argc, &argv);
  MPI_Initialized(&flag);  // check
  if (!flag) {
    std::cout << "Error: MPI_Init";
  }

  // Description of the communicator
  // communicator manages groups of parallel processes
  // determining the number of processes in a group
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  // determining the rank of a process in a group
  MPI_Comm_rank(MPI_COMM_WORLD, &myId);

  double *plBlock;
  size_t blockSize = vectorSize / numProcs;
  plBlock = new double[blockSize];

  double *plBuffer;
  plBuffer = new double[vectorSize];

  if (myId == 0) {
    for (unsigned i = 0; i < vectorSize; i++) {
      lnVector[i] = static_cast<double>(std::rand() % 50) /
                   (static_cast<double>(std::rand() % 100) + 1) -
                    static_cast<double>(std::rand() % 50) /
                   (static_cast<double>(std::rand() % 100) + 1);
      plVector[i] = lnVector[i];
    }

    // SHELL
    std::cout << std::endl << "*LINE*";

    if (vectorSize <= 10) GetVectorInfo(lnVector, vectorSize);

    lnStartTime = MPI_Wtime();
    SortShell(lnVector, vectorSize);
    lnEndTime = MPI_Wtime();
    lnWorkTime = lnEndTime - lnStartTime;

    std::cout << std::endl << "ShellSortTime: " << lnWorkTime << std::endl;

    if (vectorSize <= 10) GetVectorInfo(lnVector, vectorSize);
    // END SHELL
  }

  MPI_Barrier(MPI_COMM_WORLD);

  plStartTime = MPI_Wtime();

  MPI_Scatter(plVector, blockSize, MPI_DOUBLE, plBlock, blockSize, MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  SortShell(plBlock, blockSize);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(plBlock, blockSize, MPI_DOUBLE, plVector, blockSize, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  if (myId == 0) {
    SortMerge(plVector, plBuffer, 0, vectorSize - 1);
    plEndTime = MPI_Wtime();
    plWorkTime = plEndTime - plStartTime;

    std::cout << std::endl << "*PARALLEL*";
    std::cout << std::endl << "ShellMergeSortTime: " << plWorkTime << std::endl;

    if (Test(lnVector, plVector, vectorSize)) {
      if (vectorSize <= 10) GetVectorInfo(plVector, vectorSize);
      std::cout << std::endl << "Result: plVector" << std::endl;
    }

    if (Test(lnVector, plBuffer, vectorSize)) {
      if (vectorSize <= 10) GetVectorInfo(plBuffer, vectorSize);
      std::cout << std::endl << "Result: plBuffer" << std::endl;
    }

    std::cout << std::endl
              << "SPEED-UP: " << lnWorkTime / plWorkTime << std::endl;
  }

  MPI_Finalize();

  delete[] plVector;
  delete[] lnVector;
  delete[] plBuffer;
  delete[] plBlock;

  return 0;
}
