set terminal postscript enhanced color
set output './images/benchmark3_x0_x1_step0.eps'
set style line 1 linecolor rgb "blue"
set autoscale
unset label
set xtic auto
set ytic auto
set xlabel "x0"
set ylabel "x1"
plot '-' notitle with lines ls 1
e
