set terminal postscript enhanced color
set output 'B9_s1.eps'
set style line 1 linecolor rgb "blue"
set autoscale
unset label
set xtic auto
set ytic auto
set xlabel "x0"
set ylabel "x1"
plot '-' notitle with lines ls 1
