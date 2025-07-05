#ifndef KMEANS_H_
#define KMEANS_H_

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "mpi.h"

double getDistance(const double *x1, const double *x2, int m);
void scaling(double* const x, const int n, const int m);
void blockFunction1(const double* const x, double* const Ex, double* const Exx, const int m, const int perProc);
void blockFunction2(double* const x, const double* const Ex, const double* const Exx, const int m, const int perProc);
void MPI_Scaling(double* const x, const int n, const int m);
int getCluster(const double* const x, const double* const c, const int m, int k);
char constr(const int *y, const int val, int s);
double* getStartCores(const double* const x, const int n, const int m, const int k);
void MPI_Distrdata(const double* const x, const int m, const int numOfProc, const int perProc, int **y_in, double **x_in);
void detStartPartition(const double* const x, const double* const c, int* const y, int* const nums, int n, const int m, const int k);
void MPI_Detstartpartition(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, const double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc);
void sumCores(const double* const x, double* const c, const int* const y, const int n, const int m);
void MPI_Sumcores(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc);
void coresDiv(double* const c, const int* const nums, const int m, const int k);
void MPI_Coresdiv(double* const c, const int* const nums, const int m, const int k);
void MPI_Calccores(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc);
char checkPartition(const double* const x, const double* const c, int* const y, int* const nums, const int n, const int m, const int k);
char MPI_Checkpartition(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc);
char cyclicrecalc(const double* const x, int* const y, double* const c, int* const nums, const int n, const int m, const int k);
char MPI_Cyclicrecalc(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc);
int* kmeans_serial(const double* const x, const int n, const int m, const int k);
int* MPI_Kmeans(const double* const x, const int n, const int m, const int k);

#endif
