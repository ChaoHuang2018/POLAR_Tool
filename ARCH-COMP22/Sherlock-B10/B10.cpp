#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = "controllerB_POLAR";
	string benchmark_name = "B10_" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 6;

//	intervalNumPrecision = 600;

	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
    int x3_id = vars.declareVar("x3");
	int u0_id = vars.declareVar("u0");
    int u1_id = vars.declareVar("u1");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    vector<string> ode_list;
    ode_list.push_back("x3*cos(x2)");
    ode_list.push_back("x3*sin(x2)");
    ode_list.push_back("u1");
    ode_list.push_back("u0");
    ode_list.push_back("0");
    ode_list.push_back("0");

    ODE<Real> dynamics(ode_list, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 3;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.05, order);

	// time horizon for a single control step
//	setting.setTime(0.2);

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
	double w = 0.005;
	int steps = 50;
	Interval init_x0(9.505 - w, 9.545 + w), init_x1(-4.495 - w, -4.455 + w), init_x2(2.105 - w, 2.105 + w), init_x3(1.505 - w, 1.505 + w), init_u0(0), init_u1(0); //w=0.005
	std::vector<Interval> X0;
	X0.push_back(init_x0);
	X0.push_back(init_x1);
	X0.push_back(init_x2);
    X0.push_back(init_x3);
	X0.push_back(init_u0);
    X0.push_back(init_u1);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 1000);

	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "controllerB_POLAR";
	NeuralNetwork nn(nn_name);

	unsigned int bernstein_order = 2;
	unsigned int partition_num = 10;

	unsigned int if_symbo = 1;

	double err_max = 0;


	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

	clock_t begin, end;
	begin = clock();

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
        
//        cout << "111" << endl;

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
        
        // the normalization of the nn output will be done by adding an offset of the txt file
        // need to be double checked
//        cout << "222" << endl;
        tmv_output.tms[0].remainder = tmv_output.tms[0].remainder + Interval(-0.0001,0.0001);
//        cout << "333" << endl;

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
	Constraint c1("x0 - 0.6", vars);
	Constraint c2("-x0 - 0.6", vars);
	Constraint c3("x1 - 0.2", vars);
	Constraint c4("-x1 - 0.2", vars);
    Constraint c5("x2 - 0.06", vars);
    Constraint c6("-x2 - 0.06", vars);
    Constraint c7("x3 - 0.3", vars);
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
		printf("The target set is reachable.\n");
	}
	else
	{
		printf("The target set is NOT reachable.\n");
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);


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

	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_x0_x1", result.tmv_flowpipes, setting);
    
    plot_setting.setOutputDims("x2", "x3");
    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_x2_x3", result.tmv_flowpipes, setting);

	return 0;
}
