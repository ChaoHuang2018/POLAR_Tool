#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = argv[6];
	string benchmark_name = "B10_" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 6;

	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
	int u0_id = vars.declareVar("u0");
    int u0_id = vars.declareVar("u1");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    vector<string> ode_list;
    ode_list.push_back("x3*cos(x2)");
    ode_list.push_back("x3*sin(x2)");
    ode_list.push_back("u1");
    ode_list.push_back("u0");

    ODE<Real> dynamics(s.ode_list, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(stod(argv[7]), order);

	// time horizon for a single control step
//	setting.setTime(0.2);

	// cutoff threshold
	setting.setCutoffThreshold(1e-10);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stod(argv[1]);
	int steps = stoi(argv[2]);
	Interval init_x0(9.505 - w, 9.545 + w), init_x1(-4.495 - w, -4.455 + w), init_x2(2.105 - w, 2.105 + w), init_u0(0), init_u1(0); //w=0.005
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "B10_"+net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-10, 1e-10);
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;

	unsigned int if_symbo = stoi(argv[5]);

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	// perform 35 control steps
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);

		// TaylorModelVec<Real> tmv_temp;
		// initial_set.compose(tmv_temp, order, cutoff_threshold);
		// tmv_input.tms.push_back(tmv_temp.tms[0]);
		// tmv_input.tms.push_back(tmv_temp.tms[1]);


		// taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		TaylorModelVec<Real> tmv_output;

		if(if_symbo == 0){
			// not using symbolic remainder
			nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else{
			// using symbolic remainder
			nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}


		Matrix<Interval> rm1(1, 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;
        
        // the normalization of the nn output will be done by adding an offset of the txt file
        // need to be double checked
        tmv_output.tms[1].remainder = tmv_output.tms[1].remainder + Interval(-0.0001,0.0001);


		initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[1];


		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, initial_set, 0.2, setting, unsafeSet, symbolic_remainder);

		if (result.isCompleted())
		{
			initial_set = result.fp_end_of_time;
//			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}


	vector<Constraint> targetSet;
	Constraint c1("x0 - 0.6", vars);		// x0 <= -0.15
	Constraint c2("-x0 - 0.6", vars);		// x0 >= -0.43
	Constraint c3("x1 - 0.2", vars);		// x1 <= 0.22
	Constraint c4("-x1 - 0.2", vars);		// x1 >= 0.05
    Constraint c5("x2 - 0.06", vars);        // x1 <= 0.22
    Constraint c6("-x2 - 0.06", vars);
    Constraint c7("x3 - 0.3", vars);        // x1 <= 0.22
    Constraint c8("-x3 - 0.3", vars);

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);
    targetSet.push_back(c5);
    targetSet.push_back(c6);
    targetSet.push_back(c7);
    targetSet.push_back(c8);

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	if(b)
	{
		reach_result = "Verification result: Yes(" + to_string(steps) + ")";
	}
	else
	{
		reach_result = "Verification result: No(" + to_string(steps) + ")";
	}


	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("x0", "x1");

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/" + benchmark_name + "_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result);

	return 0;
}
