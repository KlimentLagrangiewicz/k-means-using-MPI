#include "help.h"


void fscanfData(const char* const fn, double* const x, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Error in opening %s file\n", fn);
		exit(1);
	}
	int i = 0;
	while (i < n && !feof(fl)) {
		if (fscanf(fl, "%lf", x + i) == 0) {}
		++i;
	}
	fclose(fl);
}

void fscanfPartition(const char* const fn, int* const y, const int n) {
	FILE *fl = fopen(fn, "r");
	if (!fl) {
		printf("Can't access %s file with ideal partition for reading\n", fn);
		exit(1);
	}
	int i = 0;
	while (i < n && !feof(fl)) {
		if (fscanf(fl, "%d", y + i) == 0) {
			printf("Error in reading the perfect partition from %s file\n", fn);
			exit(1);
		}
		++i;
	}
	fclose(fl);
}

double getPrecision(const int* const x, const int* const y, const int n) {
	unsigned long yy = 0ul, ny = 0ul;
	int i, j;
	for (i = 0; i < n; ++i) {
		const int xi = x[i], yi = y[i];
		for (j = i + 1; j < n; ++j) {
			if (xi == x[j] && yi == y[j]) ++yy;
			if (xi != x[j] && yi == y[j]) ++ny;
		}
	}
	return yy == 0ul && ny == 0ul ? 0.0 : (double)yy / (double)(yy + ny);
}

void fprintfTime(const char* const fn, const double t) {
	FILE *fl = fopen(fn, "a");
	if (!fl) {
		printf("Error in opening %s file\n", fn);
		exit(1);
	}
	fprintf(fl, "%lf\n", t);
	fclose(fl);
}

void fprintfResults(const char* const fn, const int* const res, const int n, const int m, const int k) {
	FILE *fl = fopen(fn, "w");
	if (!fl) {
		printf("Error in opening %s file\n", fn);
		exit(1);
	}
	fprintf(fl, "Results of clustering using k-means\nParameters: n = %d, m = %d, k = %d\ni, y_i\n", n, m, k);
	int i = 0;
	while (i < n) {
		fprintf(fl, "%d, %d\n", i, res[i]);
		++i;
	}
	fclose(fl);
}
