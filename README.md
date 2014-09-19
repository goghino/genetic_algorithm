### Implementation of GA for approximation of noisy data by 3rd order polynomial function.

(c) Juraj Kardos, 2014

Noisy data are generated by given polynomial function defined in file generator.c and adding small noise. Data are located in file input.txt. Format of file is couple x and f'(x) on each line, where f'(x) is noisy polynomial function we want to approximate.

Usage: `$./generator N`, where N is number of generated data points.

Fitness is squared sum of difference between approximation function g(x) and noisy data points. The lower fitness is the better approximation was found. Fitness = sum for 1..N(sqr(g(x\_i)-f'(x\_i)))

Exact solution, i.e. generating polynomial function f(x) without noise has these parameters:

```
$ cat solution.txt 
c0 = -5;
c1 = 3;
c2 = 4;
c3 = -2;
```

This repository consists of three different impmlementations:
1. CPU serial code
2. GPU accelerated code
3. MPI code using multiple GPUs
    a. Whole population is cloned to all GPUs, problem is solved without any
       communication during computation. At the end, each process sends
       it solution to master, who select the best one.
    b. Population is distributed across processes. During each time step
       population is scattered, mutation and evaluation of fitness is done
       locally on each process and afterwards population is gathered on master,
       where selection and crossover is performed.

Ad 1)

```
$ ./cpu input.txt 
Reading file - success!
------------------------------------------------------------
Finished! Found Solution:
	c0 = -4.78963
	c1 = 2.92177
	c2 = 3.53095
	c3 = -1.72443
Best fitness: 4.30823
Generations: 1500
Time for CPU calculation equals 35.92 seconds [16x smaller population than GPU]
```

Ad 2)

```
$ ./gpu input.txt 
Reading file - success!
------------------------------------------------------------
Finished! Found Solution:
	c0 = -5.00663
	c1 = 2.97804
	c2 = 4.04249
	c3 = -2.00143
Best fitness: 1.91558
Generations: 1500
Time for GPU calculation equals 4.21 seconds [coalesced gl. memory access]
//Time for GPU calculation equals 5.46 seconds [non-coalesced gl. memory access]
```

Ad 3.a)

```
jkardos@tesla-cmc:~/certification/genetic_algorithm/code$ mpirun -np 4 ./mpi input.txt
Reading file - success!
Reading file - success!
Reading file - success!
Reading file - success!
Rank: 0
Best fitness: 1.91651
Generations: 1500
Time for GPU calculation equals 4.96 seconds [coalesced gl. memory access, compare Ad 2.)]
Rank: 3
Best fitness: 1.91601
Generations: 1500
Time for GPU calculation equals 6.87 seconds
Rank: 1
Best fitness: 1.9157
Generations: 1500
Time for GPU calculation equals 6.78 seconds
Rank: 2
Best fitness: 1.91584
Generations: 1500
Time for GPU calculation equals 6.75 seconds
------------------------------------------------------------
Finished! Found Solution at process #1: 
	c0 = -5.00659
	c1 = 2.98028
	c2 = 4.042
	c3 = -2.00204
Best fitness: 1.9157
Generations: 1500
Time for GPU calculation equals 6.78 seconds
Time for communication equals 1.47 seconds
```

Ad 3.b)

```
jkardos@tesla-cmc:~/certification/genetic_algorithm/code$ make multirun
/home/jkardos/openmpi-1.8.1/install/bin/mpirun -np 4 /home/jkardos/certification/genetic_algorithm/code/multi /home/jkardos/certification/genetic_algorithm/code/input.txt
Reading file - success!
Reading file - success!
Reading file - success!
Reading file - success!
------------------------------------------------------------
Finished! Found Solution: 
	c0 = -5.00467
	c1 = 2.97352
	c2 = 4.04092
	c3 = -1.99897
Best fitness: 1.91614
Generations: 1500
Time for GPU calculation + communication equals 6.56 seconds
```

