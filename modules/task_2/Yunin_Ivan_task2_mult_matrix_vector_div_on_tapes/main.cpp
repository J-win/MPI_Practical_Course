// Copyright  2018 Ivan Yunin
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <iostream>


/* Description
Task 2 "Matrix vector multiplication (Division on tapes)"
The task is to multiply matrix by vector
Matrix has a dim [n x m]
Vector has a dim [m x 1]
Result of multiply is vector with dim [n x 1]

(*) Each element of result vector is scalar multiplication vector by row in matrix 

    Parallelism
    
Quantity processes = number_process

Matrix divided on tapes 
Quantity rows in tape is  k = n / number_process;    (rounded up)
Each process receives part of matrix with dim [k x m]
Each process calculate  multiplication vector by  PART of matrix
Result of work process is part of result vector

    Variables
col_num -> m; row_num -> n; sub_row_num-> k
proc_num -> number_process
flag_out -> if true - show serial result vector and parallel result vector

    Functions
check() - Checking for vector equality
scal_mult() - Scalar multiplication vector by other vector 
*/

int check(double* first, double* second, int size) {
    int res = 0;
    for (int i = 0; i < size; i++) {
        if (fabs(first[i]-second[i]) > 0.0000001) {
            res = 1;
            break;
        }
    }
    return res;
}


double scal_mult(double* a, double* b, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) {
        res+=a[i]*b[i];
    }
    return res;
}


int main(int argc, char*argv[]) {
    int col_num = 100, row_num = 100, sub_row_num;
    int proc_num, proc_id, flag;
    int flag_out = 0;
    double *vector = NULL;
    double *matrix = NULL;
    double *sub_matrix = NULL;
    double *serial_res = NULL;
    double *parallel_res = NULL;
    double *sub_parallel_res = NULL;
    double s_time_start = 0.0, p_time_start = 0.0;
    double s_time_finish = 0.0, p_time_finish = 0.0;
    std::srand(static_cast<int>(time(0)));

    if (argc > 2) {
        row_num = atoi(argv[1]);
        col_num = atoi(argv[2]);
    }
    if (argc > 3) {
        flag_out = atoi(argv[3]);
    }
    MPI_Init(&argc, &argv);
    MPI_Initialized(&flag);
    if (!flag) {
        std::cout << "Init MPI Error";
        return 0;
    }

    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    sub_row_num = static_cast<int>(ceil(static_cast<double>(row_num)/
                                        (static_cast<double>(proc_num))));
    // quantity rows for one process

// memory alloc for all processes
    vector = new double[col_num];
    for (int i = 0; i < col_num; i++)
        vector[i] = 0.0;
    sub_matrix = new double[sub_row_num*col_num];
    for (int i = 0; i < sub_row_num*col_num; i++)
        sub_matrix[i] = 0.0;

    sub_parallel_res = new double[sub_row_num];
    for (int i = 0; i < sub_row_num; i++)
        sub_parallel_res[i] = 0.0;
//  end memory alloc for all processes
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_id == 0) {
        int tail = proc_num*sub_row_num-row_num;
        /* tail-> if the quantity rows is divided between the processes 
is not entirely (for correct work Gather() )*/

        // init vector and matrix (memory and values)
        matrix = new double[row_num*col_num];
        for (int i = 0; i < row_num*col_num; i++)
            matrix[i] = 0.0;

        serial_res = new double[row_num];
        for (int i = 0; i < row_num; i++)
            serial_res[i] = 0.0;

        parallel_res = new double[row_num+tail];
        for (int i = 0; i < row_num+tail; i++)
            parallel_res[i] = 0.0;

        for (int i = 0; i < col_num; i++) {
            vector[i] = -(std::rand()%100) + (std::rand()%200)/13.0;
            for (int j = 0; j < row_num; j++) {
                matrix[j*col_num+i] = ((std::rand()%300)/17.0)-
                                      (std::rand()%100)/3.0;
            }
        }
        // end init vector and matrix (memory and values)
        // serial multiplication
        s_time_start = MPI_Wtime();
        for (int i = 0; i < row_num; i++) {
            serial_res[i] = scal_mult(vector, matrix+col_num*i, col_num);
            // (*) look description
        }
        s_time_finish = MPI_Wtime();
        std::cout << "serial time: " << s_time_finish - s_time_start << '\n';
        // serial multiplication end
    }
    MPI_Barrier(MPI_COMM_WORLD);
    p_time_start = MPI_Wtime();
    // (#)(start) code executed by all processes
    MPI_Bcast(vector, col_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast send vector to all procceses
    MPI_Scatter(matrix, sub_row_num*col_num, MPI_DOUBLE,
       sub_matrix, sub_row_num*col_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* send parts of matrix to all processes (each procces gets matrix
    with dim [sub_row_num x col_num])*/
    // begin calculate sub result
    for (int i = 0; i < sub_row_num; i++) {
            sub_parallel_res[i] = scal_mult(vector, (sub_matrix+col_num*i),
                                            col_num);  // (*) look description
        }
    // end calculate sub result
    MPI_Gather(sub_parallel_res, sub_row_num, MPI_DOUBLE, parallel_res,
               sub_row_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* collection result data in proc#0 (each procces send 
    part of result_vector, part had dim [sub_row_num x 1])*/

    // (#)(finish) code executed by all processes
    MPI_Barrier(MPI_COMM_WORLD);
    p_time_finish = MPI_Wtime();
    if (proc_id == 0) {
        if (flag_out) {
            std::cout << "Serial result vector" << '\n';
            for (int i = 0; i < row_num; i++) {
                std::cout << serial_res[i] << " ";
            }
            std::cout << '\n' << "Parallel result vector" << '\n';
            for (int i = 0; i < row_num; i++) {
                std::cout << parallel_res[i] << " ";
            }
        }

        std::cout << '\n' << "Parralel time: "
            << p_time_finish - p_time_start  << '\n';
        if (check(serial_res, parallel_res, row_num))
            std::cout << "Error: vectors are not equal" << '\n';
    }
    if ( matrix           != NULL ) { delete[]matrix;           }
    if ( serial_res       != NULL ) { delete[]serial_res;       }
    if ( parallel_res     != NULL ) { delete[]parallel_res;     }
    if ( sub_parallel_res != NULL ) { delete[]sub_parallel_res; }
    if ( vector           != NULL ) { delete[]vector;           }
    if ( sub_matrix       != NULL ) { delete[]sub_matrix;       }
    MPI_Finalize();
    return 0;
}
