//  Copyright: (c) Pahandrovich
#include <mpi.h>
#include <assert.h>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <cmath>

void quicksort(double *mas, int begin, int end);
int check_of_sort(double *mas, int size);
void mergesort(double *masresult, double *mas1, double *mas2,
           int firstElemMas1, int sizemas1, int firstElemMas2, int sizemas2);

int main(int argc, char* argv[]) {
  std::srand(static_cast<int>(time(NULL)));
  int rank, CountP, flag;
  if (argc < 2) {
    std::cout << "Error";
    return -1;
  }
  int sizemas = atoi(argv[1]);
  if (sizemas <= 0 || sizemas <= 0) {
    std::cout << "Error";
    return -1;
  }
  //  ------------ MPI Init -----------------
  MPI_Init(&argc, &argv);
  MPI_Initialized(&flag);
  if (!flag) {
    std::cout << "Error";
    return -1;
  }
  MPI_Comm_size(MPI_COMM_WORLD, &CountP);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double *mas = nullptr;
  double *accessorymas = nullptr;
  double *recievemas = nullptr;
  double *masForLine = nullptr;
  double *submas = nullptr;
  double Time_begin = 0;
  double Time_end   = 0;
  int submit_num = static_cast<int>(ceil(sizemas / CountP));

  if (rank == 0) {
    std::cout << "sizemas = " << sizemas << std::endl;
    std::cout << "CountP = " << CountP << std::endl;
    std::cout << "submit_num = " << submit_num << std::endl;
    mas = new double[sizemas];
    recievemas = new double[sizemas];
    accessorymas = new double[2*sizemas];
    masForLine = new double[sizemas];
    //  ------------ Initialization -----------------
    for (int i = 0; i < sizemas; i++)
      mas[i] = masForLine[i] = std::rand()%10000;
    //  ------------ Print mas ------------------------
    if (sizemas < 21) {
      std::cout << "array: ";
      for (int i = 0; i < sizemas; i++)
      std::cout << " " << mas[i];
    }
    std::cout << std::endl;
    //  -------------- Line ---------------------------
    Time_begin = MPI_Wtime();

    quicksort(masForLine, 0, sizemas-1);

    Time_end = MPI_Wtime();
    //  ------------ Print Line result ------------------------
    std::cout << "Line_Time = " << Time_end - Time_begin << std::endl;
    std::cout << "sorting check Line = " <<
      check_of_sort(masForLine, sizemas) << std::endl;
    if (sizemas < 21) {
      std::cout << "Line_Result: ";
      for (int i = 0; i < sizemas; i++)
      std::cout << masForLine[i] << "  ";
    }
    std::cout << std::endl;
  }

  //  ----------------- Parallel --------------------------------
  submas = new double[submit_num];

  MPI_Barrier(MPI_COMM_WORLD);
  Time_begin = MPI_Wtime();

  //  ---------------- Send ---------------------------
  MPI_Scatter(mas, submit_num, MPI_DOUBLE,
  submas, submit_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  //  ---------------- Work ----------------------------------
  quicksort(submas, 0, submit_num - 1);

  //  union result -------------------------------
  MPI_Gather(submas, submit_num, MPI_DOUBLE,
    recievemas, submit_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < submit_num; i++)
      accessorymas[i] = accessorymas[sizemas + i] = recievemas[i];

    int l = 0;
    for (l = 0; l < CountP - 1; l++)
        mergesort(accessorymas + (l % 2) * sizemas,
        accessorymas,
        recievemas,
        (!(l % 2)) * sizemas,
        (l + 1)*submit_num,
        (l + 1)*submit_num,
        submit_num);

    Time_end = MPI_Wtime();
    //  ------------ end of parallel ------------------
    for (int j = 0; j < sizemas; j++)
      mas[j] = accessorymas[(l-1)%2*sizemas + j];
    std::cout << std::endl << "Parallel time: ";
    std::cout << Time_end - Time_begin << std::endl;
    std::cout << "sorting check Parallel = " <<
          check_of_sort(mas, sizemas) << std::endl;
    if (sizemas < 21) {
      std::cout << "Parallel_Result:  ";
      for (int i = 0; i < sizemas; i++)
        std::cout << mas[i] << "  ";
    }
    std::cout << std::endl;
  }

  if (mas != nullptr) delete[]mas;
  if (masForLine != nullptr) delete[]masForLine;
  if (submas != nullptr) delete[]submas;
  if (recievemas != nullptr) delete[]recievemas;
  if (accessorymas != nullptr) delete[]accessorymas;

  MPI_Finalize();

  return 0;
}

void quicksort(double *mas, int begin, int end) {
  double middle = 0, tmp = 0;
  int a = begin, b = end;
  middle = mas[begin + (end - begin) / 2];
  while (a <= b) {
    while (mas[a] < middle) a++;
    while (mas[b] > middle) b--;
    if (a <= b) {
      if (mas[a] > mas[b]) {
        tmp = mas[a];
        mas[a] = mas[b];
        mas[b] = tmp;
      }
      a++;
      if (b > 0)
        b--;
    }
  }
  if (a < end) quicksort(mas, a, end);
  if (begin < b) quicksort(mas, begin, b);
}

int check_of_sort(double *mas, int size) {
  int i = 0;
  while (i + 1 < size && mas[i] <= mas[i + 1]) i++;
  if (i == size - 1)
    return 1;
  else
    return 0;
}

void mergesort(double *masresult, double *mas1, double *mas2,
               int firstElemMas1, int sizemas1,
               int firstElemMas2, int sizemas2) {
  int i = firstElemMas1, j = firstElemMas2, k = 0;;
  double tmp1 = 0, tmp2 = 0;
  while (i < firstElemMas1 + sizemas1 && j < firstElemMas2 + sizemas2) {
    tmp1 = mas1[i];
    tmp2 = mas2[j];
    if (tmp1 <= tmp2) {
      masresult[k] = tmp1;
      i++;
    } else {
      masresult[k] = tmp2;
      j++;
    }
    k++;
  }
  while (i < firstElemMas1 + sizemas1)
    masresult[k++] = mas1[i++];
  while (j < firstElemMas2 + sizemas2)
    masresult[k++] = mas2[j++];
}
