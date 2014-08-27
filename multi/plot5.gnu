set terminal png size 1920,1080
set output "plot5.png"
set xrange [1:4]
set yrange [100:-2000]
set pointsize 1.5

f(x) = -2*x**5 + 1*x**4 -2*x**3 + 4*x**2 + 3*x - 5


c5g = -1.9
c4g = 0.14
c3g = 1.45
c2g = -0.1
c1g = 0.32
c0g = -0.015
g_gpu(x) = c5g*x**5 + c4g*x**4 + c3g*x**3 + c2g*x**2 + c1g*x + c0g

plot "input5.txt" title "Noisy data", \
f(x) title "generating function f(x)", \
g_gpu(x) title "g(x)"
