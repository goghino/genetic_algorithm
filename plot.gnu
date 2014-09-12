set terminal png size 1920,1080
set output "plot.png"
set xrange [-2:3]
set yrange [-6:5]
set pointsize 1.5

f(x) = -2*x**3 + 4*x**2 + 3*x - 5

#solution: -2 4 3 -5
c3c = -1.72
c2c = 3.53
c1c = 2.92
c0c = -4.78
g_cpu(x) = c3c*x**3 + c2c*x**2 + c1c*x + c0c

c3g = -2.0
c2g = 4.04
c1g = 2.97
c0g = -5.00
g_gpu(x) = c3g*x**3 + c2g*x**2 + c1g*x + c0g

c3m = -2.0
c2m = 4.04
c1m = 2.98
c0m = -5.00
g_mpi(x) = c3m*x**3 + c2m*x**2 + c1m*x + c0m

plot "input.txt" title "Noisy data", \
f(x) title "generating function f(x)", \
g_cpu(x) title "CPU approximated g(x)",\
g_gpu(x) title "GPU approximated g(x)",\
g_mpi(x) title "MPI+GPUs approximated g(x)"
