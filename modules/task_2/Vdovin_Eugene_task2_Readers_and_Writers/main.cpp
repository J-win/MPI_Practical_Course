// copyright : (C) by J-win
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdlib>

int converter_in_number(const std::string &s) {
  int len = s.length();
  int a = 0;
  int i = 0;
  for (i = 0; (i < len); i++)
    a = a * 10 + (s[i] - '0');
  return a;
}

int main(int argc, char *argv[]) {
  srand(static_cast<int>(time(0)));
  int ProcNum, ProcRank;
  int storage = 0;
  int activeReaders = 0;
  int writeLock = 0;
  int readLock = 0;
  int w = 0;
  std::string s = argv[1];
  int iter = converter_in_number(s);
  int end = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Status status;
  if (ProcRank == 0) {
    int finish = 0;
    while (end) {
      int flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&writeLock, 1, MPI_INT, w, 1, MPI_COMM_WORLD);
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &flag, &status);
      if (flag)
MPI_Recv(&writeLock, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &flag, &status);
      if (flag)
MPI_Recv(&readLock, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
        MPI_Send(&activeReaders, 1, MPI_INT, w, 5, MPI_COMM_WORLD);
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, &flag, &status);
      if (flag)
MPI_Recv(&storage, 1, MPI_INT, MPI_ANY_SOURCE, 6, MPI_COMM_WORLD, &status);
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 7, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 7, MPI_COMM_WORLD, &status);
        MPI_Send(&readLock, 1, MPI_INT, w, 8, MPI_COMM_WORLD);
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 9, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 9, MPI_COMM_WORLD, &status);
        activeReaders++;
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 10, MPI_COMM_WORLD, &status);
        MPI_Send(&storage, 1, MPI_INT, w, 11, MPI_COMM_WORLD);
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 12, MPI_COMM_WORLD, &status);
        activeReaders--;
      }
      flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &flag, &status);
      if (flag) {
        MPI_Recv(&w, 1, MPI_INT, MPI_ANY_SOURCE, 15, MPI_COMM_WORLD, &status);
        finish++;
      }
      if (finish == ProcNum - 1) {
        std::cout << "Work end" << std::endl;
        end = 0;
      }
    }
  } else {
    int k = ProcRank % 2;
    switch (k) {
    case 0: {
      int writerNumber = ProcRank;
      bool flag1, flag2;
      for (int i = 0; i < iter; i++) {
        flag1 = flag2 = false;
        int timeSleep = std::rand() * writerNumber % 15 + 3;
        double beg = MPI_Wtime();
        while (MPI_Wtime() - beg < timeSleep) {}
        std::cout << ": Writer " << writerNumber;
        std::cout << " turned to storage" << std::endl;
        do {
          MPI_Send(&ProcRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          MPI_Recv(&writeLock, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
          if (writeLock == 1) {
            if (!flag2) {
                std::cout << "Another writer turned to storage before.";
                std::cout << " Writer ";
                std::cout << writerNumber << " expects" << std::endl;
            }
            flag2 = true;
            beg = MPI_Wtime();
            while (MPI_Wtime() - beg < 0.1) {}
          }
        } while (writeLock == 1);
        writeLock = 1;
        MPI_Send(&writeLock, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        readLock = 1;
        MPI_Send(&readLock, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        do {
          MPI_Send(&ProcRank, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
          MPI_Recv(&activeReaders, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
          if (activeReaders != 0) {
            if (!flag1) {
              std::cout << "Writer " << writerNumber << " expects, ";
              std::cout << "bye active readers completed ";
              std::cout << "with storage" << std::endl;
            }
            flag1 = true;
            beg = MPI_Wtime();
            while (MPI_Wtime() - beg < 0.1) {}
          }
        } while (activeReaders != 0);
        std::cout << "Writer " << writerNumber;
        std::cout << " gained access to the storage" << std::endl;
        beg = MPI_Wtime();
        while (MPI_Wtime() - beg < 3) {}
        storage = (std::rand() + writerNumber) % 401 + 100;
        MPI_Send(&storage, 1, MPI_INT, 0, 6, MPI_COMM_WORLD);
        std::cout << "Writer " << writerNumber << " write in storage ";
        std::cout << storage;
        std::cout << " and completed work" << std::endl;
        readLock = 0;
        MPI_Send(&readLock, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        writeLock = 0;
        MPI_Send(&writeLock, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
      }
      MPI_Send(&ProcRank, 1, MPI_INT, 0, 15, MPI_COMM_WORLD);
      break;
    }
    case 1: {
      int readerNumber = ProcRank;
      bool flag;
      for (int i = 0; i < iter; i++) {
        flag = false;
        int timeSleep = std::rand() * readerNumber % 12 + 3;
        double beg = MPI_Wtime();
        while (MPI_Wtime() - beg < timeSleep) {}
        std::cout << ": Reader " << readerNumber;
        std::cout << " turned to storage" << std::endl;
        do {
          MPI_Send(&ProcRank, 1, MPI_INT, 0, 7, MPI_COMM_WORLD);
          MPI_Recv(&readLock, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);
          if (readLock == 1) {
            if (!flag) {
              std::cout << "Access to the storage is blocked. Reader ";
              std::cout << readerNumber << " expects" << std::endl;
            }
            flag = true;
            beg = MPI_Wtime();
            while (MPI_Wtime() - beg < 0.1) {}
          }
        } while (readLock == 1);
        std::cout << ": Reader " << readerNumber;
        std::cout << " gained access to the storage" << std::endl;
        MPI_Send(&activeReaders, 1, MPI_INT, 0, 9, MPI_COMM_WORLD);
        MPI_Send(&ProcRank, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
        MPI_Recv(&storage, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &status);
        beg = MPI_Wtime();
        while (MPI_Wtime() - beg < 3) {}
        std::cout << ": Reader " << readerNumber;
        std::cout << " read from storage ";
        std::cout << storage << " and completed work" << std::endl;
        MPI_Send(&activeReaders, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
      }
      MPI_Send(&ProcRank, 1, MPI_INT, 0, 15, MPI_COMM_WORLD);
      break;
    }
    }
  }
  MPI_Finalize();
  return 0;
}
