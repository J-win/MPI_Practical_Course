/* Copyright Nikita Danilov */
#include <mpi.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <algorithm>

#include "include/host.h"

host::host(int rank, int size) {
  id = rank;
  proc_number = size;
}

host::~host() {
}

template<class T>
const T& minself(const T& a, const T& b) {
    return (a < b) ? a : b;
}

void host::generate_packet(int src, int dst, std::string msg) {
  size_t len = minself<size_t>(MAX_DATA_SIZE - 1, msg.length());

  pkt.src = src;
  pkt.dst = dst;
  for (size_t i = 0; i < len; i++)
    pkt.data[i] = msg[i];
  pkt.data[len] = '\0';
}

MPI_Datatype register_mpi_type(packet const&pkt) {
  const int num_members = 3;
  int lengths[num_members] = { 1, 1, MAX_DATA_SIZE};
  MPI_Aint offsets[num_members] = { offsetof(packet, src),
                                    offsetof(packet, dst),
                                    offsetof(packet, data) };
  MPI_Datatype types[num_members] = { MPI_INT, MPI_INT, MPI_CHAR };

  MPI_Datatype struct_type;
  MPI_Type_create_struct(num_members, lengths, offsets, types, &struct_type);
  MPI_Type_commit(&struct_type);
  return struct_type;
}

void deregister_mpi_type(MPI_Datatype type) {
  MPI_Type_free(&type);
}

ring_host::ring_host(int rank, int size) : host(rank, size) {
  pkt.src = -1;
  pkt.dst = -1;

  for (size_t i = 0; i < MAX_DATA_SIZE - 1; ++i)
    pkt.data[i] = ' ';
  pkt.data[MAX_DATA_SIZE - 1] = '\0';
}

ring_host::~ring_host() {
}

int ring_host::who_next() const {
  int res;

  if (id == pkt.dst)
    return id;

  if (std::abs(pkt.dst - pkt.src) > proc_number / 2) {
    if (pkt.dst - pkt.src > 0)
      res = ((id == 0) ? proc_number - 1 : id - 1);
    else
      res = (id + 1) % proc_number;
  } else {
    if (pkt.dst - pkt.src > 0)
      res = (id + 1) % proc_number;
    else
      res = ((id == 0) ? proc_number - 1 : id - 1);
  }

  return res;
}

void ring_host::xmit() {
  int done = 0, dest;
  MPI_Status status;

  type = register_mpi_type(pkt);
  std::ios_base::sync_with_stdio(false);

  while (done != 1) {
    if (pkt.src != -1) {
      dest = who_next();
      for (int dst_id = 0; dst_id < proc_number; ++dst_id)
        if (dst_id != id)
          MPI_Send(&dest, 1, MPI_INT, dst_id, id, MPI_COMM_WORLD);

      if (dest != id) {
        MPI_Send(&pkt, 1, type, dest, id, MPI_COMM_WORLD);
        pkt.src = -1;
        std::cout << "Process " << id << " send packet to " << dest
                << " process" << std::endl;
      }
    } else {
       MPI_Recv(&dest, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, &status);
      if (dest == id) {
        MPI_Recv(&pkt, 1, type, status.MPI_SOURCE, status.MPI_SOURCE,
                 MPI_COMM_WORLD, &status);
        std::cout << "Process " << id << " received packet from "
                  << status.MPI_SOURCE << " process" << std::endl;
      }
    }
    if (pkt.dst != -1 && id == pkt.dst) {
      done = 1;
      std::cout << std::endl << "Packet transfer completed successfully!"
                << std::endl << std::endl;
      std::cout << "Packet data:" << std::endl;
      std::cout << "src = " << pkt.src << std::endl;
      std::cout << "dst = " << pkt.dst << std::endl;
      std::cout << "msg = " << std::string(pkt.data) << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (dest == id) {
      for (int dst_id = 0; dst_id < proc_number; ++dst_id)
        if (dst_id != id)
          MPI_Send(&done, 1, MPI_INT, dst_id, id, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
               MPI_COMM_WORLD, &status);
    }
  }

  deregister_mpi_type(type);
}
