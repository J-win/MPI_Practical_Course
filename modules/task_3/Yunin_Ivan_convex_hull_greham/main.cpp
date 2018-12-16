    // Copyright  2018 Ivan Yunin
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>

struct Point{
    int x;
    int y;
};

// global because it is used in compare function)
Point p0;
void my_swap(Point &p1, Point &p2) {
    Point temp = p1;
    p1 = p2;
    p2 = temp;
}


// Checks whether the line is crossing the polygon
int orientation(Point a, Point b,
                Point c) {
    int res = (b.y - a.y) * (c.x - b.x) -
              (c.y - b.y) * (b.x - a.x);

    if (res == 0) {
        return 0;
    }
    if (res > 0) {
        return 1;
    }
    return -1;
}

int distance(Point p1, Point p2) {
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

// compare function for sorting
int compare(const void *vp1, const void *vp2) {
    Point *p1 = reinterpret_cast<Point*>(&vp1);
    Point *p2 = reinterpret_cast<Point*>(&vp2);
    int o = orientation(p0, *p1, *p2);
    if (o == 0) {
        if (distance(p0, *p2) >= distance(p0, *p1)) {
            return -1;
        } else {
            return 1;
        }
    }
    if (o == -1) {
        return -1;
    } else {
        return 1;
    }
}

// Finds upper tangent of two polygons 'a' and 'b'
// represented as two vectors. //rerurn size of vector
int merger(Point *a, Point *b, Point *out, int size_a, int size_b) {
    int n1 = size_a, n2 = size_b;
    int ia = 0, ib = 0;
    for (int i = 1; i < n1; i++)
        if (a[i].x > a[ia].x)
            ia = i;

    // ib -> leftmost point of b
    for (int i = 1; i < n2; i++)
        if (b[i].x < b[ib].x)
            ib = i;

    // finding the upper tangent
    int inda = ia, indb = ib;
    bool done = 0;
    while (!done) {
        done = 1;
        while (orientation(b[indb], a[inda],
                           a[(inda+1)%n1]) >= 0)
            inda = (inda + 1) % n1;

        while (orientation(a[inda], b[indb],
                           b[(n2+indb-1)%n2]) <= 0) {
            indb = (n2+indb-1)%n2;
            done = 0;
        }
    }

    int uppera = inda, upperb = indb;
    inda = ia, indb = ib;
    done = 0;
    while (!done) {
        done = 1;
        while (orientation(a[inda], b[indb], b[(indb+1)%n2]) >= 0)
            indb = (indb+1)%n2;

        while (orientation(b[indb], a[inda], a[(n1+inda-1)%n1]) <= 0) {
            inda = (n1+inda-1)%n1;
            done = 0;
        }
    }

    int lowera = inda, lowerb = indb;

    int ind = uppera;
    int out_size = -1;
    out[++out_size] = a[uppera];
    while (ind != lowera) {
        ind = (ind+1)%n1;
        out[++out_size] = a[ind];
    }

    ind = lowerb;
    out[++out_size] = b[lowerb];
    while (ind != upperb) {
        ind = (ind+1)%n2;
        out[++out_size] = b[ind];
    }
    return out_size+1;
}


int convex_hull(Point *points, Point *out, int size) {
    int ymin = points[0].y, min = 0;
    int n = size;
    for (int i = 1; i < n; i++) {
       int y = points[i].y;
       if ((y < ymin) || (ymin == y &&
           points[i].x < points[min].x))
       ymin = points[i].y, min = i;
    }
    my_swap(points[0], points[min]);
//  std::cout << "begin:\n";
    p0 = points[0];
    qsort(&points[1], n-1, sizeof(Point), compare);
//   std::cout << "after qsort:\n";
    int m = 1;
    for (int i = 1; i < n; i++) {
        while (i < n-1 && orientation(p0, points[i],
                                    points[i+1]) == 0)
        i++;
        points[m] = points[i];
        m++;
    }
//  std::cout << "afrer eql points:\n";
    if (m < 3) return 0;
    int out_size = -1;
    out[++out_size] = points[0];
    out[++out_size] = points[1];
    out[++out_size] = points[2];
    for (int i = 3; i < m; i++) {
        while (orientation(out[out_size-1], out[out_size], points[i]) != -1)
            --out_size;
        out[++out_size] = points[i];
    }
//  std::cout << "after main while:\n";
    return out_size+1;
}

int compare_X(const void *vp1, const void *vp2) {
    Point *p1 = reinterpret_cast<Point *>(&vp1);
    Point *p2 = reinterpret_cast<Point *>(&vp2);
    return ( p1->x - p2->x );
}

void init_map(Point * res, int n, int u_bound, int l_bound) {
    Point p;
    for (int i = 0; i < n; i++) {
        p.y = 1+std::rand()%(u_bound-1);
        p.x = 1+std::rand()%(l_bound-1);
        res[i] = p;
    }
}

int main(int argc, char*argv[]) {
    srand(static_cast<int>(time(0)));
    int num_p = 100;
    int proc_num, proc_id, flag, sub_num_p, seq_res_size;
    double s_time_start = 0.0, p_time_start = 0.0;
    double s_time_finish = 0.0, p_time_finish = 0.0;
    Point *points = NULL, *seq_res = NULL, *par_res = NULL,
    *sub_points = NULL, *sub_res = NULL, *points2 = NULL;
    Point *for_merge[2];
    Point for_disp;
    MPI_Datatype PNT;
    MPI_Datatype type[2] = { MPI_INT, MPI_INT };
    MPI_Aint disp[2];
    MPI_Aint addresses[3];
    int blocklen[2] = { 1, 1 };
    if (argc > 1) {
        num_p = atoi(argv[1]);
    }
    MPI_Init(&argc, &argv);
    MPI_Initialized(&flag);
    if (!flag) {
        std::cout << "Init MPI Error";
        return 0;
    }
    MPI_Get_address(&(for_disp), &addresses[0]);
    MPI_Get_address(&(for_disp.x), &addresses[1]);
    MPI_Get_address(&(for_disp.y), &addresses[2]);
    disp[0] = addresses[1] - addresses[0];
    disp[1] = addresses[2] - addresses[0];

    MPI_Type_create_struct(2, blocklen, disp, type, &PNT);
    MPI_Type_commit(&PNT);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    // static_cast<double>
    sub_num_p = num_p/proc_num;
    int *sub_res_size = new int[proc_num];
    sub_points = new Point[sub_num_p];
    sub_res = new Point[sub_num_p];
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_id == 0) {
        points = new Point[num_p];
        points2 = new Point[num_p];
        seq_res = new Point[num_p];
        par_res = new Point[num_p];
        init_map(points, num_p, num_p, num_p);
        for (int i = 0; i < num_p; i++)
        points2[i] = points[i];
        s_time_start = MPI_Wtime();
        seq_res_size = convex_hull(points, seq_res, num_p);
        s_time_finish = MPI_Wtime();
        qsort(&points2[0], num_p, sizeof(Point), compare_X);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    p_time_start = MPI_Wtime();
    MPI_Scatter(points2 , sub_num_p, PNT, sub_points,
    sub_num_p, PNT, 0, MPI_COMM_WORLD);
    sub_res_size[proc_id] = convex_hull(sub_points, sub_res, sub_num_p);
    MPI_Gather(sub_res_size+proc_id, 1, MPI_INT, sub_res_size, 1,
               MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(sub_res, sub_num_p, PNT, par_res, sub_num_p, PNT, 0,
               MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (proc_id == 0) {
        int shift = 0;
        int for_merge_size[2]={ num_p, num_p };
        for_merge[0] = new Point[num_p];
        for_merge_size[0] =  merger(par_res, par_res+sub_num_p, for_merge[0],
                                    sub_res_size[0], sub_res_size[1]);
        for_merge[1] = new Point[num_p];
        shift += sub_num_p;
        for (int i = 1; i <proc_num-1; i++) {
            shift+=sub_num_p;
            for_merge_size[i%2] = merger(for_merge[(i+1)%2], par_res + shift,
                                         for_merge[i%2],
                                         for_merge_size[(i+1)%2],
                                         sub_res_size[i+1]);
            flag = i%2;
        }
        p_time_finish = MPI_Wtime();
        std::cout << "seq " << s_time_finish - s_time_start << std::endl;
        std::cout << "par " << p_time_finish - p_time_start << std::endl;
    }
    if ( points     != NULL ) { delete[]points;     }
    if ( seq_res    != NULL ) { delete[]seq_res;    }
    if ( par_res    != NULL ) { delete[]par_res;    }
    if ( sub_points != NULL ) { delete[]sub_points; }
    if ( sub_res    != NULL ) { delete[]sub_res;    }
    if ( points2    != NULL ) { delete[]points2;    }
    MPI_Finalize();
    return 0;
}
