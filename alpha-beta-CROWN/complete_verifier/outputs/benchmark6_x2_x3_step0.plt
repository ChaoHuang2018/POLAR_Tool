set terminal postscript enhanced color
set output './images/benchmark6_x2_x3_step0.eps'
set style line 1 linecolor rgb "blue"
set autoscale
unset label
set xtic auto
set ytic auto
set xlabel "x2"
set ylabel "x3"
plot '-' notitle with lines ls 1
e
