#include "../../POLAR/NNCS.h"
//#include "../../flowstar/flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	
	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int u_id = vars.declareVar("u");

	vector<string> derivatives = {"-x0+x1-x2","-x0*(x2+1)-x1","-x0+u","0"};

	string net_name = argv[6];
	string benchmark_name = "nncs_reachnn_benchmark_4_" + net_name;
	string nn_name = "nn_4_"+net_name;
	NeuralNetwork nn_controller(nn_name);
	//NeuralNetwork nn_controller("nn_1_sigmoid");


	NNCS<Real> system(vars, 0.1, derivatives, nn_controller);


	// Flow* reachability setting
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);
	//unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(stod(argv[7]), order);
	//setting.setFixedStepsize(0.1, order);
	setting.setCutoffThreshold(1e-10);
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(vars.size(), I);
	setting.setRemainderEstimation(remainder_estimation);


	// POLAR setting
	unsigned int taylor_order = order;	// same as the flowpipe order, but it is not necessary
	unsigned int bernstein_order = stoi(argv[3]);
	//unsigned int bernstein_order = 3;
	unsigned int partition_num = 4000;
	//unsigned int partition_num = 10;
	unsigned int if_symbo = stoi(argv[5]);

	PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Symbolic");

	if(if_symbo == 0){
			// not using symbolic remainder
			polar_setting.set_remainder_type("Concrete");
		}

	// initial set
	double w = stod(argv[1]);
	//double w = 0.05;

	// define the initial set which is a box
	Interval init_x0(0.26 - w, 0.26 + w), init_x1(0.09 - w, 0.09 + w), init_x2(0.26 - w, 0.26 + w);

	// the initial values of the rest of the state variables are 0
	vector<Interval> box(vars.size());
	box[x0_id] = init_x0;
	box[x1_id] = init_x1;
	box[x2_id] = init_x2;

	Flowpipe initialSet(box);


	// unsafe set
	vector<Constraint> safeSet;

	Result_of_Reachability result;

	// run the reachability computation
	clock_t begin, end;
	double seconds;
	begin = clock();

	int n = stoi(argv[2]); // time horizon
	//double n = 500; // time horizon

	Symbolic_Remainder sr(initialSet, 1000);

	system.reach(result, initialSet, n, setting, polar_setting, safeSet, sr);

	

	vector<Constraint> targetSet;
	if (net_name == "relu" || net_name == "relu_tanh")
    {
        Constraint c1("x0 + 0.1", vars);		// x0 <= -0.1
        Constraint c2("-x0 - 0.2", vars);		// x0 >= -0.2
        Constraint c3("x1 - 0.05", vars);		// x1 <= 0.05
        Constraint c4("-x1", vars);		        // x1 >= 0.0

        targetSet.push_back(c1);
        targetSet.push_back(c2);
        targetSet.push_back(c3);
        targetSet.push_back(c4);
    }
    else
    {
        Constraint c1("x0 - 0.05", vars);		// x0 <= 0.05
        Constraint c2("-x0 - 0.05", vars);		// x0 >= -0.05
        Constraint c3("x1", vars);		        // x1 <= 0.0
        Constraint c4("-x1 - 0.05", vars);		// x1 >= -0.05

        targetSet.push_back(c1);
        targetSet.push_back(c2);
        targetSet.push_back(c3);
        targetSet.push_back(c4);
    }
	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(n) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(n) + ")";
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
	
	seconds = (end - begin) / CLOCKS_PER_SEC;
	std::string running_time ="Running Time: %lf\n" + to_string(seconds) + " seconds";


	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.printOn();
	plot_setting.setOutputDims("x0", "x1");
	
	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(n) + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);

	
	return 0;
}
