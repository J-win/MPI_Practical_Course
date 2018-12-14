#include <iostream>
#include <mpi.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <assert.h>

using namespace std;
double max_func(double* first, int size){
    double max = first[0];
    for(int i = 0; i<size;i++){
    if (first[i]>max)
        max = first[i];
    }
    return max;
}

int main (int argc, char* argv[]){
    int tail_flag=0; //for сheck the existence  vector "tail"
    int size=42, proc_num,proc_id, block_size,flag;
    double *vector, *max_vector,*temp_vector;
    double serial_time, parallel_time;
    double serial_max, parallel_max, temp_max;
    MPI_Status status;
    srand((int)time(0));

    if (argc>1){
        size = atoi(argv[1]);
    }

    vector = new double[size];

    for(int i = 0; i<size; i++){
        vector[i]=-20 + (rand()%200)/13.0;
    }
    MPI_Init(&argc, &argv);
    MPI_Initialized(&flag);
    if (!flag)
    {
        cout << "Error";
        delete[]vector;
        return 0;
    }
    MPI_Comm_size(MPI_COMM_WORLD,&proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD,&proc_id);
    block_size = size/proc_num;
    if(size%proc_num!=0){
                tail_flag=1;
    }//then last process is not active
    if(block_size!=0){ // if proc_num is more than size, exit the program
        if (proc_id == 0){
            //serial max search
            serial_time = MPI_Wtime();
            serial_max = max_func(vector,size);
            serial_time = MPI_Wtime() - serial_time;
            cout<<"serial  max: "<<serial_max<<endl;
            cout<<"serial time: "<<serial_time<<endl;
            //serial max search end
            parallel_time = MPI_Wtime();
            max_vector = new double[proc_num-1+tail_flag];
            for (int i = 1;i<proc_num-1;i++){
                MPI_Send(vector+block_size*i,block_size,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
            }
            if(tail_flag){
                MPI_Send(vector+block_size*proc_num,size-block_size*proc_num,MPI_DOUBLE,proc_num-1,0,MPI_COMM_WORLD);
            }
            max_vector[0] = max_func(vector,block_size);
            for (int i = 1;i<proc_num-1+tail_flag;i++){
                MPI_Recv(&max_vector[i],1,MPI_DOUBLE,i,0,MPI_COMM_WORLD, &status);
            }
            parallel_max = max_func(max_vector,proc_num-1+tail_flag);
            parallel_time = MPI_Wtime() - parallel_time;
            cout<<"Parralel max: "<<parallel_max<<endl;
            cout<<"Parralel time: "<<parallel_time<<endl;
            delete[]max_vector;
        }
        else{
            if( proc_id != proc_num-1){
                temp_vector = new double[block_size];
                MPI_Recv(temp_vector,block_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD, &status);
                temp_max = max_func(temp_vector,block_size);
                delete[]temp_vector;
                MPI_Send(&temp_max,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
            }
            else{
                if(tail_flag){
                int tail_size = size-block_size*proc_num;
                temp_vector = new double[tail_size];
                MPI_Recv(temp_vector,block_size,MPI_DOUBLE,0,0,MPI_COMM_WORLD, &status);
                temp_max = max_func(temp_vector,tail_size);
                delete[]temp_vector;
                MPI_Send(&temp_max,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
                }
            }
        }
    }
    else{
        MPI_Finalize();
        cout<<"Vector size < number of processes.";
        delete[]vector;
        return 0;
    }
    MPI_Finalize();
    delete[]vector;
    return 0;
}