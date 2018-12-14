/* Copyright Nikita Danilov */
#include <assert.h>
#include <time.h>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <string>

#include "include/host.h"

int main(int argc, char* argv[]) {
  /* Mpi variables */
  int rank,
      size;
  int source, destination;
  std::string msg;

  if (argc > 3) {
    source = atoi(argv[1]);
    destination =  atoi(argv[2]);
    msg = std::string(argv[3]);
  } else {
    source = -1;
    destination = -1;
    msg = std::string("Hi");
  }

  /* Mpi init block */
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ring_host *test_host = new ring_host(rank, size);

  if (source == rank) {
    test_host->generate_packet(source, destination, msg);
  }


  MPI_Barrier(MPI_COMM_WORLD);
  test_host->xmit();

  MPI_Finalize();

  return 0;
}
