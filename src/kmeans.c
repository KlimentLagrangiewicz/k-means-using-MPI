#include "kmeans.h"


double getDistance(const double *x1, const double *x2, int m) {
	double d, r = 0.0;
	for (; m > 0; --m) {
		d = *x1 - *x2;
		r += d * d;
		++x1;
		++x2;
	}
	return r;
}

void scaling(double* const x, const int n, const int m) {
	const double* const end = x + n * m;
	int j;
	for (j = 0; j < m; ++j) {
		double sd, Ex = 0.0, Exx = 0.0, *ptr;
		for (ptr = x + j; ptr < end; ptr += m) {
			sd = *ptr;
			Ex += sd;
			Exx += sd * sd;
		}
		Exx /= n;
		Ex /= n;
		sd = Exx - Ex * Ex;
		if (sd == 0.0) sd = 1.0;
		else sd = 1.0 / sqrt(sd);
		for (ptr = x + j; ptr < end; ptr += m) {
			*ptr = (*ptr - Ex) * sd;
		}
	}
}

void blockFunction1(const double* const x, double* const Ex, double* const Exx, const int m, const int perProc) {
	int i, j;
	for (i = 0; i < perProc * m; i += m) { 
		const double* const x_i = x + i;
		for (j = 0; j < m; ++j) {
			Ex[j] += x_i[j];
			Exx[j] += x_i[j] * x_i[j];
		}
	}
}

void blockFunction2(double* const x, const double* const Ex, const double* const Exx, const int m, const int perProc) {
	int i, j;
	for (i = 0; i < perProc * m; i += m)
		for (j = 0; j < m; ++j) 
			x[i + j] = (x[i + j] - Ex[j]) * Exx[j];
}

void MPI_Scaling(double* const x, const int n, const int m) {
	int numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	int perProc = n / numOfProc;
	if (perProc == 0) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) scaling(x, n, m);
	} else {
		double *localX = (double*)malloc(perProc * m * sizeof(double));
		MPI_Scatter(x, perProc * m, MPI_DOUBLE, localX, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		double *Ex = (double*)calloc(m, sizeof(double));
		double *Exx = (double*)calloc(m, sizeof(double));
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0 && n > perProc * numOfProc) blockFunction1(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		blockFunction1(localX, Ex, Exx, m, perProc);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Ex, Ex, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) MPI_Reduce(MPI_IN_PLACE, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		else MPI_Reduce(Exx, Exx, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		if (pid == 0) {
			int i;
			for (i = 0; i < m; ++i) {
				Ex[i] /= n;
				Exx[i] /= n;
				Exx[i] = Exx[i] - Ex[i] * Ex[i];
				if (Exx[i] == 0.0) Exx[i] = 1.0;
				Exx[i] = 1.0 / sqrt(Exx[i]);
			}
		}
		MPI_Bcast(Ex, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(Exx, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (pid == 0 && n > perProc * numOfProc) blockFunction2(x + perProc * numOfProc * m, Ex, Exx, m, n - perProc * numOfProc);
		blockFunction2(localX, Ex, Exx, m, perProc);
		MPI_Gather(localX, perProc * m, MPI_DOUBLE, x, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(Ex);
		free(Exx);
		free(localX);
	}
}


int getCluster(const double* const x, const double* const c, const int m, int k) {
	int res = --k;
	double min_d = getDistance(x, c + k * m, m);
	while (k > 0) {
		--k;
		const double cur_d = getDistance(x, c + k * m, m);
		if (cur_d < min_d) {
			min_d = cur_d;
			res = k;
		}
	}
	return res;
}

char constr(const int *y, const int val, int s) {
	for (; s > 0; --s) {
		if (*y == val) return 1;
		++y;
	}
	return 0;
}

double* getStartCores(const double* const x, const int n, const int m, const int k) {
	double *c = (double*)malloc(k * m * sizeof(double));
	int *nums = (int*)malloc(k * sizeof(int));
	srand((unsigned int)time(NULL));
	int i;
	for (i = 0; i < k; ++i) {
		int val = rand() % n;
		while (constr(nums, val, i)) val = rand() % n;
		
		nums[i] = val;
		memcpy(c + i * m, x + val * m, m * sizeof(double));
	}
	free(nums);
	return c;
}




void MPI_Distrdata(const double* const x, const int m, const int numOfProc, const int perProc, int **y_in, double **x_in) {
	if (numOfProc == 1 || perProc == 0) {
		*y_in = NULL;
		*x_in = NULL;
		return;
	}
	double *localX = (double*)malloc(perProc * m * sizeof(double));
	MPI_Scatter(x, perProc * m, MPI_DOUBLE, localX, perProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	int *localY = (int*)malloc(perProc * sizeof(int));
	
	*y_in = localY;
	*x_in = localX;
}

void detStartPartition(const double* const x, const double* const c, int* const y, int* const nums, int n, const int m, const int k) {
	while (n > 0) {
		--n;
		const int l = getCluster(x + n * m, c, m, k);
		y[n] = l;
		++nums[l];
	}
}

void MPI_Detstartpartition(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, const double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc) {
	memset(nums, 0, k * sizeof(int));
	detStartPartition(x_local, c, y_local, nums, perProc, m, k);
	if (n > numOfProc * perProc) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			const int buf = numOfProc * perProc;
			detStartPartition(x_global + buf * m, c, y_global + buf, nums, n - buf, m, k);
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, nums, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

void sumCores(const double* const x, double* const c, const int* const y, const int n, const int m) {
	int i, j;
	for (i = 0; i < n; ++i) {
		double* const c_yi = c + y[i] * m;
		const double* const x_i = x + i * m;
		for (j = 0; j < m; ++j) {
			c_yi[j] += x_i[j];
		}
	}
}

void MPI_Sumcores(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc) {
	memset(c, 0, k * m * sizeof(double));
	sumCores(x_local, c, y_local, perProc, m);
	if (n > numOfProc * perProc) {
		int pid;
		MPI_Comm_rank(MPI_COMM_WORLD, &pid);
		if (pid == 0) {
			const int buf = numOfProc * perProc;
			sumCores(x_global + buf * m, c, y_global + buf, n - buf, m);
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, c, k * m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void coresDiv(double* const c, const int* const nums, const int m, const int k) {
	int i, j;
	for (i = 0; i < k; ++i) {
		const double f = nums[i] == 0 ? 1.0 : 1.0 / nums[i];
		double* const c_i = c + i * m;
		for (j = 0; j < m; ++j) {
			c_i[j] *= f;
		}
	}
}

void MPI_Coresdiv(double* const c, const int* const nums, const int m, const int k) {
	int numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	int perProc = k / numOfProc;
	int pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	if (perProc == 0) {
		if (pid == 0) coresDiv(c, nums, m, k);
		MPI_Bcast(c, k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	} else {
		if (pid == 0 && k > numOfProc * perProc) coresDiv(c + perProc * numOfProc * m, nums + perProc * numOfProc, m, k - perProc * numOfProc);
		coresDiv(c + pid * perProc * m, nums + perProc * pid, m, perProc);
		MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, c, perProc * m, MPI_DOUBLE, MPI_COMM_WORLD);
		if (k > numOfProc * perProc) MPI_Bcast(c + perProc * numOfProc * m, k * m - perProc * numOfProc * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
}

void MPI_Calccores(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc) {
	MPI_Sumcores(x_local, y_local, x_global, y_global, c, nums, n, m, k, numOfProc, perProc);
	coresDiv(c, nums, m, k);
}

char checkPartition(const double* const x, const double* const c, int* const y, int* const nums, const int n, const int m, const int k) {
	char flag = 0;
	int i; 
	for (i = 0; i < n; ++i) {
		const int f = getCluster(x + i * m, c, m, k);
		if (y[i] != f) flag = 1;
		y[i] = f;
		++nums[f];
	}
	return flag;
}

char MPI_Checkpartition(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc) {
	memset(nums, 0, k * sizeof(int));
	char flag = checkPartition(x_local, c, y_local, nums, perProc, m, k);
	
	int pid;
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);	
	if (pid == 0 && n > numOfProc * perProc) flag |= checkPartition(x_global + numOfProc * perProc * m, c, y_global + numOfProc * perProc, nums, n - numOfProc * perProc, m, k);
	
	MPI_Allreduce(MPI_IN_PLACE, nums, k, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_CHAR, MPI_BOR, MPI_COMM_WORLD);
	
	return flag;
}

char cyclicrecalc(const double* const x, int* const y, double* const c, int* const nums, const int n, const int m, const int k) {
	memset(c, 0, k * m * sizeof(double));
	sumCores(x, c, y, n, m);
	coresDiv(c, nums, m, k);
	
	memset(nums, 0, k * sizeof(int));
	return checkPartition(x, c, y, nums, n, m, k);
}

char MPI_Cyclicrecalc(const double* const x_local, int* const y_local, const double* const x_global, int* const y_global, double* const c, int* const nums, const int n, const int m, const int k, const int numOfProc, const int perProc) {
	MPI_Calccores(x_local, y_local, x_global, y_global, c, nums, n, m, k, numOfProc, perProc);
	
	return MPI_Checkpartition(x_local, y_local, x_global, y_global, c, nums, n, m, k, numOfProc, perProc);
}

int* kmeans_serial(const double* const x, const int n, const int m, const int k) {
	double *c = getStartCores(x, n, m, k);
	int *y = (int*)malloc(n * sizeof(int));
	int *nums = (int*)malloc(k * sizeof(int));
	memset(nums, 0, k * sizeof(int));
	
	detStartPartition(x, c, y, nums, n, m, k);
	
	while(cyclicrecalc(x, y, c, nums, n, m, k));
	
	free(c);
	free(nums);
	
	return y;
}


int* MPI_Kmeans(const double* const x, const int n, const int m, const int k) {
	double *x_local;
	int *y_local;
	int pid, numOfProc;
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	int perProc = n / numOfProc;
	if (perProc == 0 || numOfProc == 1) {
		int *y = pid == 0 ? kmeans_serial(x, n, m, k) : (int*)malloc(n * sizeof(int));
		
		MPI_Bcast(y, n, MPI_INT, 0, MPI_COMM_WORLD);
		return y;
	}	
	MPI_Distrdata(x, m, numOfProc, perProc, &y_local, &x_local);
	
	double *c = pid == 0 ? getStartCores(x, n, m, k) : (double*)malloc(k * m * sizeof(double));
	MPI_Bcast(c, k * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	int *y = (int*)malloc(n * sizeof(int));
	int *nums = (int*)malloc(k * sizeof(int));
	MPI_Detstartpartition(x_local, y_local, x, y, c, nums, n, m, k, numOfProc, perProc);
	
	while (MPI_Cyclicrecalc(x_local, y_local, x, y, c, nums, n, m, k, numOfProc, perProc));
	
	free(c);
	free(nums);
	free(x_local);	
	
	MPI_Gather(y_local, perProc, MPI_INT, y, perProc, MPI_INT, 0, MPI_COMM_WORLD);
	free(y_local);
	MPI_Bcast(y, n, MPI_INT, 0, MPI_COMM_WORLD);
	return y;
}