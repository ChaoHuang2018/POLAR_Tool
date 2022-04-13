clear;
for i = 0:12
    for k = 0:10000
%         f = "xy_step_" + int2str(i) + "_" + int2str(k);
        f = "yz_step_" + int2str(i) + "_" + int2str(k);
        if ~isfile(f+".m")
            break;
        end
        disp(f+";");
    end
end
clear;