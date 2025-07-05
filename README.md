# k-means-using-MPI
Implementation of k-means clustering algorithm in C programming language using MPI technology

## About k-means
The k-means algorithm is an iterative data clustering algorithm developed by Stuart Lloyd of Bell Labs in the 1950s as a technique for pulse-code modulation.  
The main idea of the algorithm is that at each iteration, based on the existing partitioning, the cluster centers are recalculated, then the instances are divided into clusters according to which of the new centers turned out to be closer to a specific instance according to a pre-selected metric.  
#### Input data:
  +  $X=\mathrm{x}_{i=1,j=1}^{n,m}$ — description of instances, where *n* is number of instances, *m* is number of features;  
  +  $k \in \left \\{1, \ldots, n \right \\}$ — number of clusters.  
#### Output data:
  +  $Y = \left\\{ y_i|y_i\in\left\\{0,\ldots,k-1\right\\}, i = \overline{\left(1,n\right)}\right\\}$ — cluster labels.  
#### Advantages of k-means:
  +  Low algorithmic complexity;  
  +  Easy to implement;  
  +  The possibility for effective parallelization;  
  +  The presence of many modifications.  
#### Disadvantages of k-means:
  +  Sensitivity to initial cluster centers;  
  +  Algorithm k-means poorly separates closely spaced clusters with a complex structure;  
  +  The need for preliminary determination of the number of clusters.  
### Steps of k-means algorithm:
Step 1. Data preparing (autoscaling): $x_{i,j}=\frac{x_{i,j}-\mathrm{E_{X^{j}}}}{\sigma_{X^{j}}}$;  
Step 2. Set initial cluster centers: $C = \left\\{c_{i}| c_{i} \in ℝ^{m}, i = \overline{\left(1,k\right)} \right\\}$;  
Step 3. Calculate the initial partition: $y_{i} = \arg\min\limits_{j}\rho\left(x_{i},c_{j} \right)$;  
Step 4. Calculate new cluster centers:  
$$l_j=\sum_{i=1}^{n}{\left[y_i\equiv j\right]}$$  
$$c_j=\frac{1}{l_j} \sum_{i=1}^{n}{\left[y_i\equiv j\right]\cdot x_i}$$  
Step 5. Calculate a new split: $y_{i} = \arg\min\limits_{j}\rho\left(x_{i},c_{j} \right)$;  
Step 6. Repeat steps 4, 5 until the split changes.  
A visualisation of the first four iterations of the algorithm is shown in figure 1.  
<p align="center"> <img width="500" height="500" src="https://github.com/KlimentLagrangiewicz/k-means-in-C/assets/81409101/c91edbf3-5c59-4a41-b6d9-e3f57f0c6516"> </p>  
<p align="center">Figure 1 — Processing the Old Faithful Geyser dataset using the k-means algorithm</p>  

## Example of usage
Cloning project and changing current directory:
```
git clone https://github.com/KlimentLagrangiewicz/k-means-using-MPI
cd k-means-using-MPI
```
Building from source (Linux):
```
make
```
Building from source (Windows):
```
make windows
```
If building was successfully, you can find executable file in `bin` subdirectory.  
Run the program:
```
mpiexec -n 7 ./bin/mpi-kmeans ./datasets/iris/data.txt 150 4 3 ./datasets/iris/new_result.txt ./datasets/iris/res.txt
```
```
mpiexec --oversubscribe -n 16 ./bin/mpi-kmeans ./datasets/iris/data.txt 150 4 3 ./datasets/iris/new_result.txt ./datasets/iris/res.txt
```