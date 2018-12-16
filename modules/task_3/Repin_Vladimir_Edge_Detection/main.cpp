// Copyright mezotaken
#include <assert.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

const char* argKeys =
"{ i image       | <none> |  Image name                }"
"{ k kernel      | 3      |  Kernel size = 1,3,5,7     }"
"{ y height      | 1024   |  Image height              }"
"{ x width       | 1024   |  Image width               }"
"{ s save        | false  |  Save image                }";


int main(int argc, char** argv) {
  cv::CommandLineParser parser(argc, argv, argKeys);
  int status = 0, procId = 0, nProc = 0;  // MPI vars
  double t = 0;  // For time
  int nRow = 0, nCol = 0;  // Size of image
  int tasksize = 0;    // Aux for calc counts
  int shift = 0;      // Aux for calc displs
  int* sendcounts = NULL, *recvcounts = NULL;   // Number of rows for process
  int* senddispls = NULL, *recvdispls = NULL;   // Displs of rows for process
  cv::Mat original;      // Original image
  cv::Mat parres, seqres;  // Results of edge finding
  cv::Mat resgrad, grad_x, grad_y;  // Gradient Mats
  int ksize = parser.get<unsigned int>("k");  // kernel size


  status = MPI_Init(&argc, &argv);
  if (status != MPI_SUCCESS) {
    std::cout << "Error while MPI Initializing";
    return -1;
  }
  // Getting process ID
  status = MPI_Comm_rank(MPI_COMM_WORLD, &procId);
  if (status != MPI_SUCCESS) {
    std::cout << "Error while getting process ID";
    return -1;
  }
  // Getting number of processes
  status = MPI_Comm_size(MPI_COMM_WORLD, &nProc);
  if (status != MPI_SUCCESS) {
    std::cout << "Error while getting number of processes";
    return -1;
  }

  sendcounts = new int[nProc];
  senddispls = new int[nProc];
  recvcounts = new int[nProc];
  recvdispls = new int[nProc];

  if (procId == 0) {
    if (parser.has("i")) {
      original = cv::imread(parser.get<cv::String>("i"));  // Load image
      nRow = original.size[0];
      nCol = original.size[1];
    } else {
      nCol = parser.get<unsigned int>("x");
      nRow = parser.get<unsigned int>("y");
      original = cv::Mat(nRow, nCol, CV_8UC3);
      cv::randu(original, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    }
    parres = cv::Mat(nRow, nCol, CV_8U);

    tasksize = nRow / nProc;
    for (int i = 0; i < nProc; i++) {
      recvcounts[i] = tasksize;
      if (i < nRow%nProc)
        recvcounts[i]++;
      recvcounts[i] *= nCol;
      sendcounts[i] = recvcounts[i];
      if (i > 0 && i < nProc - 1)
        sendcounts[i] += (ksize-1)*nCol;
      else if (nProc > 1) sendcounts[i] += (ksize/2)*nCol;
      recvdispls[i] = shift;
      if (i != 0)
        senddispls[i] = recvdispls[i] - (ksize/2)*nCol;
      else
        senddispls[0] = 0;
      shift += recvcounts[i];
    }

    // Blur it to avoid noise
    GaussianBlur(original, original, cv::Size(3, 3), 0);
    cvtColor(original, original, CV_BGR2GRAY);  // Convert to gray
    cv::Mat grad_x, grad_y;

    // Start of sequential part
    t = MPI_Wtime();

    Sobel(original, grad_x, CV_16S, 1, 0, ksize);
    convertScaleAbs(grad_x, grad_x);
    Sobel(original, grad_y, CV_16S, 0, 1, ksize);
    convertScaleAbs(grad_y, grad_y);
    addWeighted(grad_x, 0.5, grad_y, 0.5, 0, seqres);

    t = MPI_Wtime() - t;
    // End of sequential part

    std::cout << "-----------------------------" << std::endl;
    std::cout << "Sequential Time: " << t << std::endl;
    if (parser.has("s"))
    cv::imwrite("seqres.jpg", seqres);
  }

  // Start of Parallel part
  MPI_Barrier(MPI_COMM_WORLD);
  t = MPI_Wtime();

  MPI_Bcast(&nRow, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nCol, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(recvdispls, nProc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(recvcounts, nProc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(senddispls, nProc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(sendcounts, nProc, MPI_INT, 0, MPI_COMM_WORLD);
  cv::Mat tmp(sendcounts[procId] / nCol, nCol, CV_8U);
  MPI_Scatterv(original.data, sendcounts, senddispls,
  MPI_UNSIGNED_CHAR, tmp.data, sendcounts[procId],
  MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  // Calcualting gradients in both directions
  // then approximating sqrt(dx^2+dy^2) as |dx|+|dy|
  Sobel(tmp, grad_x, CV_16S, 1, 0, ksize);
  convertScaleAbs(grad_x, grad_x);
  Sobel(tmp, grad_y, CV_16S, 0, 1, ksize);
  convertScaleAbs(grad_y, grad_y);
  addWeighted(grad_x, 0.5, grad_y, 0.5, 0, resgrad);


  uchar* aux = resgrad.data;
  if (procId > 0)
    aux += (ksize / 2)*nCol;

  MPI_Gatherv(aux, recvcounts[procId], MPI_UNSIGNED_CHAR, parres.data,
    recvcounts, recvdispls, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  // End of Parallel part
  if (procId == 0) {
    t = MPI_Wtime() - t;
    int unequal = 0;
    if (parser.has("s"))
    cv::imwrite("parres.jpg", parres);
    for (int i = 0; i < nRow*nCol; i++)
      if (seqres.data[i] != parres.data[i])
        unequal++;
    std::cout << "Parallel result " << t << std::endl;
    std::cout << "Number of wrong pixels: " << unequal << std::endl;
    std::cout << "-----------------------------" << std::endl;
  }

  delete[] sendcounts;
  delete[] recvcounts;
  delete[] senddispls;
  delete[] recvdispls;

  status = MPI_Finalize();
  return 0;
}
