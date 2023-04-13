set terminal postscript enhanced color
set output './images/benchmark4_x1_x2_step0.eps'
set style line 1 linecolor rgb "blue"
set autoscale
unset label
set xtic auto
set ytic auto
set xlabel "x1"
set ylabel "x2"
plot '-' notitle with lines ls 1
e
