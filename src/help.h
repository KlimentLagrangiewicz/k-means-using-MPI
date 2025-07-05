#ifndef HELP_H_
#define HELP_H_

#include <stdlib.h>
#include <stdio.h>


void fscanfData(const char* const fn, double* const x, const int n);
void fscanfPartition(const char* const fn, int* const y, const int n);
double getPrecision(const int* const x, const int* const y, const int n);
void fprintfTime(const char* const fn, const double t);
void fprintfResults(const char* const fn, const int* const res, const int n, const int m, const int k);

#endif
