// Copyright ivanvikhrev
#include <assert.h>
#include <mpi.h>
#include <iostream>
#include <ctime>
#include <utility>
#include <limits>

#define MainProc 0

void swp(unsigned int *a, unsigned int *b) {
    unsigned int tmp = *a;
    *a = *b;
    *b = tmp;
}

unsigned int* merge(unsigned int *arr1, unsigned int num1,
    unsigned int *arr2, unsigned int num2) {
    unsigned int *res;
    unsigned int i = 0,
        j = 0,
        index = 0;

    res = new unsigned int[num1 + num2];

    while (i < num1 && j < num2) {
        if (arr1[i] < arr2[j])
            res[index++] = arr1[i++];
        else
            res[index++] = arr2[j++];
    }

    while (i < num1)
        res[index++] = arr1[i++];

    while (j < num2)
        res[index++] = arr2[j++];
    return res;
}

void rad_sort_1(unsigned int *from, unsigned int *to, unsigned int bit) {
    if (!bit || to < from + 1) return;

    unsigned int *ll = from, *rr = to - 1;

    for (;;) {
        while (ll < rr && !(*ll & bit)) ll++;
        while (ll < rr && (*rr & bit)) rr--;
        if (ll >= rr) break;
        swp(ll, rr);
    }

    if (!(bit & *ll) && ll < to) ll++;
    bit >>= 1;

    rad_sort_1(from, ll, bit);
    rad_sort_1(ll, to, bit);
}

void rad_sort_2(unsigned int * a, unsigned int count) {
    unsigned int mIndex[4][256] = { 0 };  // count / index matrix
    unsigned int *b = new unsigned int[count];  // allocate temp array
    unsigned int i, j, m, n;
    unsigned int u;
    for (i = 0; i < count; i++) {
        u = a[i];
        for (j = 0; j < 4; j++) {
            mIndex[j][static_cast<unsigned int>(u & 0xff)]++;
            u >>= 8;
        }
    }
    for (j = 0; j < 4; j++) {  // convert to indices
        m = 0;
        for (i = 0; i < 256; i++) {
            n = mIndex[j][i];
            mIndex[j][i] = m;
            m += n;
        }
    }
    for (j = 0; j < 4; j++) {  // radix sort
        for (i = 0; i < count; i++) {  // sort by current lsb
            u = a[i];
            m = static_cast<unsigned int>(u >> (j << 3)) & 0xff;
            b[mIndex[j][m]++] = u;
        }
        std::swap(a, b);  // swap ptrs
    }
    delete[] b;
}

bool is_sorted(unsigned int* arr, unsigned int size) {
    bool flag = 1;
    for (unsigned int i = 0; i < size - 1; i++)
        if (arr[i] > arr[i + 1]) {
            flag = 0;
            break;
        }
    return flag;
}

int main(int argc, char** argv) {
    int status = 0, rank = 0, size = 0;  // MPI vars
    int *scounts = NULL,  // use in scatterv
        *displs = NULL;  // use in scatterv
    unsigned int arrSize = 0;
    unsigned int  *arr = NULL,  // data
              *buff = NULL,  // buffers for message exchanging
              *buff2 = NULL,
              *resultS = NULL,  // results
              *resultP = NULL;
    double t1 = 0, t2 = 0;
    int type_sort = 0;
    int imin = std::numeric_limits<int>::min();
    status = MPI_Init(&argc, &argv);
    if (status != MPI_SUCCESS) {
     return -1;
    }

    status = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (status != MPI_SUCCESS) {
        return -1;
    }

    status = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (status != MPI_SUCCESS) {
        return -1;
    }

    if (rank == MainProc) {
        if (argc > 1) {
            arrSize = atoi(argv[1]);
            if (argc > 2) type_sort = atoi(argv[2]);
        } else {
            arrSize = 10;
            type_sort = 0;
          }
        std::cout << "  Array size : " << arrSize << std::endl;
        arr = new unsigned int[arrSize];
        resultS = new unsigned int[arrSize];
        resultP = new unsigned int[arrSize];
        std::srand((unsigned)time(NULL));
        for (unsigned int i = 0; i < arrSize; i++)
            arr[i] = resultS[i] = std::rand() % 201;
        if (arrSize < 50) {
            for (unsigned int i = 0; i < arrSize; i++)
                std::cout << arr[i] << " ";
            std::cout << std::endl;
        }
        t1 = MPI_Wtime();
        if (type_sort == 0)
            rad_sort_1(resultS, resultS + arrSize, imin);
        else
            rad_sort_2(resultS, arrSize);
        t2 = MPI_Wtime();
        std::cout << "  Sequential Time: " << t2 - t1 << std::endl;
        if (bool f = is_sorted(resultS, arrSize))
            std::cout << "    Array is sorted, flag = " << f << std::endl;
        else
            std::cout << "    Array isn`t sorted, flag = " << f << std::endl;
        if (arrSize < 50) {
            for (unsigned int i = 0; i < arrSize; i++)
                std::cout << resultS[i] << " ";
            std::cout << std::endl;
        }
        buff = new unsigned int[arrSize / size + arrSize % size];
        buff2 = new unsigned int[arrSize / size ];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    MPI_Bcast(&arrSize, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
    MPI_Bcast(&type_sort, 1, MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);
    scounts = new int[size];
    displs = new int[size];
    displs[0] = 0;
    scounts[0] = arrSize / size + arrSize % size;
    for (int i = 1; i < size; i++) {
        scounts[i] = arrSize / size;
        displs[i] = scounts[0] + (i-1)*scounts[i];
    }
    if (rank != 0)
        buff = new unsigned int[scounts[rank]];

    MPI_Scatterv(arr, scounts, displs, MPI_UNSIGNED, buff,
        scounts[rank], MPI_UNSIGNED, MainProc, MPI_COMM_WORLD);

    if (type_sort == 0)
        rad_sort_1(buff, buff + scounts[rank], imin);
    else
        rad_sort_2(buff, scounts[rank]);

    if (rank != 0)
        MPI_Send(buff, scounts[rank], MPI_UNSIGNED, MainProc,
            rank, MPI_COMM_WORLD);


    if (rank == 0) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(scounts[0]); i++)
            resultP[i] = buff[i];
        for (unsigned int i = 1; i < static_cast<unsigned int>(size); i++) {
            MPI_Recv(buff2, arrSize / size, MPI_UNSIGNED, MPI_ANY_SOURCE,
                i, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
            resultP = merge(buff2, arrSize / size, resultP,
               scounts[0] + (i - 1)*scounts[1]);
        }
        t2 = MPI_Wtime();
        std::cout << "  Parallel Time: " << t2 - t1 << std::endl;
        if (bool f = is_sorted(resultS, arrSize))
            std::cout << "    Array is sorted, flag = " << f << std::endl;
        else
            std::cout << "    Array isn`t sorted, flag = " << f << std::endl;
        if (arrSize < 50) {
            for (unsigned int i = 0; i < arrSize; i++)
                std::cout << resultP[i] << " ";
            std::cout << std::endl;
        }
        delete[] buff2;
        delete[] resultS;
        delete[] resultP;
        delete[] arr;
    }
    delete[] buff;
    delete[] scounts;
    delete[] displs;
    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        return -1;
    }
    return 0;
}
