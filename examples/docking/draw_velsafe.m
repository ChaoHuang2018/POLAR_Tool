xs = 0:30;
ys = 0:30;
cdata = zeros(31, 31);
for i = 1:31
    for j = 1:31
        cdata(32 - i, j) = 0.2 + 2.0 * 0.001027 * sqrt(xs(i) * xs(i) + ys(j) * ys(j));
    end
end

h = heatmap(xs,flip(ys),cdata);

h.Title = 'Vsafe';
h.XLabel = 'X-Position';
h.YLabel = 'Y-Position';