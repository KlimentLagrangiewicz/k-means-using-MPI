#include <stdio.h>
#include <mpi.h>

#include "help.h"
#include "kmeans.h"

int main(int argc, char* argv[]) {
	
	MPI_Init(&argc, &argv);
	
	int pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	if (argc < 6) {
		puts("Not enough parameters...");
		if (pid == 0) MPI_Abort(MPI_COMM_WORLD, 1);
	}	
	const int n = atoi(argv[2]), m = atoi(argv[3]), k = atoi(argv[4]);
	if (n <= 0 || m <= 0 || k <= 0 || k > n) {
		puts("Value of parameters is incorrect...");
		if (pid == 0) MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	double *x = NULL;
	if (pid == 0) {
		x = (double*)malloc(n * m * sizeof(double));
		fscanfData(argv[1], x, n * m);
	}
	
	MPI_Scaling(x, n, m);
	
	double t1 = MPI_Wtime();
	int *y = MPI_Kmeans(x, n, m, k);
	t1 = MPI_Wtime() - t1;
	
	if (pid == 0) {
		printf("Time for k-means clustering with usage MPI technology: %.8lf\n", t1);
		fprintfResults(argv[5], y, n, m, k);
		if (argc > 6) {
			int *ideal = (int*)malloc(n * sizeof(int));
			fscanfPartition(argv[6], ideal, n);
			const double p = getPrecision(ideal, y, n);
			free(ideal);
			printf("Precision of k-means clustering with usage MPI technology: %lf\n", p);
		}
		free(x);
	}
	
	
	free(y);
	
	MPI_Finalize();
	
	return 0;
}
