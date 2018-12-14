// copyright : (C) by J-win
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <cstdlib>

#define pi 3.14159265358979323846

double f1(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return sin(x) + sin(10.0 * x / 3.0) + sum;
}

double f2(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return (3.0 * x - 1.4) * sin(18.0 * x) + sum;
}

double f3(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return -1.0 * (x + sin(x)) * exp(-1.0 * x * x) + sum;
}

double f4(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return sin(x) + sin(10.0 * x / 3.0) + log(x) - 0.84 * x + 3.0 + sum;
}

double f5(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return -1.0 * sin(2.0 * pi * x) * exp(-1.0 * x) + sum;
}

double f6(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return (x * x - 5.0 * x + 6.0) / (x * x + 1.0) + sum;
}

double f7(double x) {
  double sum = 0.0;
  for (int i = 1; i <= 100000; i++ )
sum += sin(sin(sin(i)))*sin(sin(sin(i))) + cos(sin(sin(i)))*cos(sin(sin(i)));
  sum -= 100000;
  return -1.0 * x + sin(3.0 * x) - 1.0 + sum;
}

double (*f[7])(double) = { f1, f2, f3, f4, f5, f6, f7 };

struct limit {
  double a;
  double b;
  double x;
  double z;
  explicit limit(double a0 = 0, double b0 = 0, double x_ = 0, double z_ = 0) {
    a = a0;
    b = b0;
    x = x_;
    z = z_;
  }
};

limit fun[7] = { limit(2.7, 7.5, 5.145735, -1.899599),
limit(0.0, 1.2, 0.96609, -1.48907),
limit(-10.0, 10.0, 0.679560, -0.824239),
limit(2.7, 7.5, 5.19978, -1.60130),
limit(0.0, 4.0, 0.224885, -0.788685),
limit(-5.0, 5.0, 2.41420, -0.03553),
limit(0.0, 6.5, 5.877287, -7.815670) };

int converter_in_number(const std::string &s) {
  int len = s.length();
  int a = 0;
  int i = 0;
  for (i = 0; (i < len); i++)
    a = a * 10 + (s[i] - '0');
  return a;
}

double rank_number(const double a, const int i) {
  double b = 1;
  for (int j = 0; j < i; j++)
    b *= a;
  return b;
}

double converter_in_number_double(const std::string &s) {
  int len = s.length();
  double a = 0.0;
  int i = 0;
  for (i = 0; ((i < len) && (s[i] != '.')); i++)
    a = a * 10.0 + (s[i] - '0');
  int j = i;
  if (s[j] == '.') {
    for (i = j + 1; i < len; i++)
      a = a + (s[i] - '0') / (rank_number(10.0, (i - j)));
  }
  return a;
}

struct point {
  double x;
  double z;
  explicit point(double x_ = 0, double z_ = 0): x(x_), z(z_) {}
};

struct interval {
  double R;
  point* lp;
  point* rp;
  explicit interval(double R_ = 0.0, point* lp_ = NULL, point* rp_ = NULL) {
    R = R_;
    lp = lp_;
    rp = rp_;
  }
  interval(const interval& i) {
    R = i.R;
    lp = i.lp;
    rp = i.rp;
  }
  interval& operator=(const interval &i) {
    R = i.R;
    lp = i.lp;
    rp = i.rp;
    return *this;
  }
};

bool operator<(const interval& i1, const interval& i2) {
  return (i1.R < i2.R) ? true : false;
}

double Rfunc(const point &lp_, const point &rp_, double m) {
  double dx = rp_.x - lp_.x;
  double dz = rp_.z - lp_.z;
  return (m * dx + dz * dz / (m * dx) - 2.0 * (rp_.z + lp_.z));
}

point* insertup_list(std::list<point> *p, point *xk) {
  std::list<point>::iterator itl, itr;
  itl = itr = (*p).begin();
  while ((itr != (*p).end()) && (itr->x < (*xk).x)) {
    itl = itr;
    itr++;
  }
  (*p).insert(itr, (*xk));
  itl++;
  return &(*itl);
}

void linAGP(int j, int n, double ee) {
  std::list<point> p;
  std::priority_queue<interval> q;
  std::list<point>::iterator itl, itr;
  interval zk;

  double m = -1.0, mm, minf, minx;
  int k = 0;
  p.push_back(point(fun[j].a, f[j](fun[j].a)));
  p.push_back(point(fun[j].b, f[j](fun[j].b)));

  do {
    double mold = m;
    mm = 0.0;
    itr = itl = p.begin();
    itr++;
    while (itr != p.end()) {
      double max = fabs((itr->z - itl->z) / (itr->x - itl->x));
      if (mm < max)
        mm = max;
      itr++;
      itl++;
    }
    if (mm > 0.0)
      m = 2.0 * mm;
    else
      m = 1.0;
    if (mold != m) {
      while (!q.empty())
        q.pop();
      itr = itl = p.begin();
      itr++;
      while (itr != p.end()) {
        q.push(interval(Rfunc(*itl, *itr, m), &(*itl), &(*itr)));
        itl++;
        itr++;
      }
    }
    zk = q.top();
    q.pop();
    double xk = 0.5*(zk.rp->x + zk.lp->x) - ((zk.rp->z - zk.lp->z)/(2.0 * m));
    point t(xk, f[j](xk));
    point* tt = insertup_list(&p, &t);
    q.push(interval(Rfunc(*zk.lp, *tt, m), zk.lp, tt));
    q.push(interval(Rfunc(*tt, *zk.rp, m), tt, zk.rp));
    k++;
  } while ((zk.rp->x - zk.lp->x > ee) && (k < n));

  itl = p.begin();
  minf = itl->z;
  minx = itl->x;
  itl++;

  while (itl != p.end()) {
    if (minf > itl->z) {
      minf = itl->z;
      minx = itl->x;
    }
    itl++;
  }
  std::cout << "Arg min f = " << minx << std::endl;
  std::cout << "Min f = " << minf << std::endl;
  std::cout << "Number iterations = " << k << std::endl;
}

int main(int argc, char *argv[]) {
  int ProcNum, ProcRank;
  int j = converter_in_number(argv[1]);
  int n = converter_in_number(argv[2]);
  double ee = converter_in_number_double(argv[3]);
  double m = -1.0, mm = 0.0, minf = 0.0, minx = 0.0;
  int k = 0;
  int end = 1;
  double smm[5];
  double tp[2];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
  MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
  MPI_Status status;

  if (ProcRank == 0) {
    double stlin = MPI_Wtime();
    linAGP(j, n, ee);
    std::cout << "Lin time work = " << MPI_Wtime() - stlin << std::endl;
    std::cout << std::endl;
    double st = MPI_Wtime();
    std::list<point> p;
    std::priority_queue<interval> q;
    std::list<point>::iterator itl, itr;
    interval *zk = new interval[ProcNum - 1];
    point* tt;

    double pr = (fun[j].b - fun[j].a) / (ProcNum - 1);
    for (int i = 0; i < ProcNum - 1; i++)
      p.push_back(point(fun[j].a + pr * i, f[j](fun[j].a + pr * i)));
    p.push_back(point(fun[j].b, f[j](fun[j].b)));

    do {
      double mold = m;
      mm = 0.0;

      itr = itl = p.begin();
      itr++;
      while (itr != p.end()) {
        double max = fabs((itr->z - itl->z) / (itr->x - itl->x));
        if (mm < max)
          mm = max;
        itr++;
        itl++;
      }
      if (mm > 0.0)
        m = 2.0 * mm;
      else
        m = 1.0;

      if (mold != m) {
        while (!q.empty())
          q.pop();
        itr = itl = p.begin();
        itr++;
        while (itr != p.end()) {
          q.push(interval(Rfunc(*itl, *itr, m), &(*itl), &(*itr)));
          itl++;
          itr++;
        }
      }
      for (int i = 0; i < ProcNum - 1; i++) {
        zk[i] = q.top();
        q.pop();
        smm[0] = zk[i].lp->x;
        smm[1] = zk[i].lp->z;
        smm[2] = zk[i].rp->x;
        smm[3] = zk[i].rp->z;
        smm[4] = m;
        MPI_Send(&smm, 5, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
      }
      for (int i = 0; i < ProcNum - 1; i++) {
        MPI_Recv(&tp, 2, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD, &status);
        point t(tp[0], tp[1]);
        tt = insertup_list(&p, &t);
        q.push(interval(Rfunc(*zk[i].lp, *tt, m), zk[i].lp, tt));
        q.push(interval(Rfunc(*tt, *zk[i].rp, m), tt, zk[i].rp));
        k++;
      }
      end = 1;
      for (int i = 0; i < ProcNum - 1; i++) {
        if (zk[i].rp->x - zk[i].lp->x <= ee)
          end = 0;
      }
      if (k >= n)
        end = 0;
      for (int i = 0; i < ProcNum - 1; i++)
        MPI_Send(&end, 1, MPI_INT, i + 1, 5, MPI_COMM_WORLD);
    } while (end != 0);
    itl = p.begin();
    minf = itl->z;
    minx = itl->x;
    itl++;

    while (itl != p.end()) {
      if (minf > itl->z) {
        minf = itl->z;
        minx = itl->x;
      }
      itl++;
    }
    std::cout << "Arg min f = " << minx << std::endl;
    std::cout << "Min f = " << minf << std::endl;
    std::cout << "Number iterations = " << k << std::endl;
    std::cout << "Parallel time work = " << MPI_Wtime() - st  << std::endl;
    std::cout << std::endl;
    std::cout << "Optimum arg min f = " << fun[j].x << std::endl;
    std::cout << "Optimum min f = " << fun[j].z << std::endl;
    delete[] zk;
  } else {
    do {
      MPI_Recv(&smm, 5, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      tp[0] = 0.5 * (smm[2] + smm[0]) - ((smm[3] - smm[1]) / (2.0 * smm[4]));
      tp[1] = f[j](tp[0]);
      MPI_Send(&tp, 2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
      MPI_Recv(&end, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
    } while (end != 0);
  }

  MPI_Finalize();
  return 0;
}
