#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = "controller_double_pendulum_more_robust_POLAR";
	string benchmark_name = "Double_Pendulum_more";
	// Declaration of the state variables.
	unsigned int numVars = 7;

//	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
    int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
    int t_id = vars.declareVar("t");
	int u0_id = vars.declareVar("u0");
    int u1_id = vars.declareVar("u1");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x2", "x3", "cos(x0 - x1)*((x2*x2)*sin(x0 - x1) - cos(x0 - x1)*(2*sin(x0) - x3*x3*sin(x0 - x1)/2+4*u0)+2*sin(x1)+8*u1)/((cos(x0 - x1)*cos(x0 - x1) - 2)) - x3*x3*sin(x0 - x1)/2+2*sin(x0)+4*u0", "-(x2*x2*sin(x0 - x1) - cos(x0 - x1)*(2*sin(x0) - x3*x3*sin(x0 - x1)/2+4*u0)+2*sin(x1)+8*u1)/(cos(x0 - x1)*cos(x0 - x1)*0.5 - 1)", "1","0","0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.002, order);

	// time horizon for a single control step
//	setting.setTime(0.5);

	// cutoff threshold
	setting.setCutoffThreshold(1e-6);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

//	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = 0;
	int steps = 5;
	Interval init_x0(1, 1.01), init_x1(1, 1.01), init_x2(1, 1.01), init_x3(1, 1.01), init_t(0), init_u0(0), init_u1(0); //w=0.01
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
    X0.push_back(init_x3);
    X0.push_back(init_t);
	X0.push_back(init_u0);
    X0.push_back(init_u1);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
    // safe set: when t>=0.4, all the veriables [-0.5, 1.5]
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-10, 1e-10);
	unsigned int bernstein_order = order;
	unsigned int partition_num = 100;

	unsigned int if_symbo = 1;

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
        tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);

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


//		Matrix<Interval> rm1(1, 1);
//		tmv_output.Remainder(rm1);
//		cout << "Neural network taylor remainder: " << rm1 << endl;


		initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[1];

		// if(if_symbo == 0){
		// 	dynamics.reach(result, setting, initial_set, unsafeSet);
		// }
		// else{
		// 	dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		// }

		// Always using symbolic remainder
		dynamics.reach(result, initial_set, 0.02, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
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
	Constraint c1("x0 - 0.2", vars);		// x0 <= 0.2
	Constraint c2("-x0 - 0.1", vars);		// x0 >= -0.1
	Constraint c3("x1 + 0.6", vars);		// x1 <= -0.6
	Constraint c4("-x1 - 0.9", vars);		// x1 >= -0.9

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);

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
	plot_setting.setOutputDims("x2", "x3");

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
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result.tmv_flowpipes, setting);

	return 0;
}
