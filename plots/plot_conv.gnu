#input parameters:
#   c3g
#   c2g
#   c1g
#   c0g
#   out - output name   
#   generation - generation ID

set terminal png size 1920,1080
set output out
set xrange [-3:4]
set yrange [-7:6]
set pointsize 1.5
set xlabel generation

set title "Convergence of the GA"

g_gpu(x) = c3g*x**3 + c2g*x**2 + c1g*x + c0g

plot "../input.txt" title "Noisy data", \
g_gpu(x) title "GPU approximated g(x)"

