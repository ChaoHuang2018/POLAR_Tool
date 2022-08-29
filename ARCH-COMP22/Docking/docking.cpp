#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	string net_name = "model_POLAR";
	string benchmark_name = "docking_" + net_name;
	// Declaration of the state variables.
	unsigned int numVars = 6;


	Variables vars;

	int x0_id = vars.declareVar("x0");
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int u0_id = vars.declareVar("u0");
    int u1_id = vars.declareVar("u1");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
    ODE<Real> dynamics({"x2","x3","0.002054*x3 + 0.000003164187*x0 + 0.083333333333333*u0","-0.002054*x2 + 0.083333333333333*u1","0","0"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.05, order);

	// time horizon for a single control step
//	setting.setTime(0.5);

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.1, 0.1);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

//	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	int steps = 40;
//	Interval init_x0(70, 70), init_x1(70, 70), init_x2(0.28, 0.28), init_x3(0.28, 0.28), init_u0(0), init_u1(0); //w=0.01
    Interval init_x0(87, 89), init_x1(87, 89), init_x2(-0.01, 0.01), init_x3(-0.01, 0.01), init_u0(0), init_u1(0);
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
	vector<Constraint> safeSet;// = {Constraint("sqrt(x2^2 + x3^2) - 0.2 - 0.002054 * sqrt(x0^2 + x1^2)", vars)};

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = net_name;
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	unsigned int bernstein_order = 2;
	unsigned int partition_num = 4000;

	unsigned int if_symbo = 1;

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
		dynamics.reach(result, initial_set, 1, setting, safeSet, symbolic_remainder);

        if(result.status == COMPLETED_UNSAFE)
        {
        	break;
        }

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
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

	if(result.isUnsafe())
	{
		printf("The system is unsafe.\n");
	}
	else if(result.isSafe())
	{
		printf("The system is safe.\n");
	}
	else
	{
		printf("The safety is unknown.\n");
	}

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);


	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
//	plot_setting.setOutputDims("sqrt(x2^2+x3^2)", "sqrt(x0^2+x1^2)");
    


	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x0", "x1");
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "docking_x0_x1", result.tmv_flowpipes, setting);
    plot_setting.setOutputDims("x2", "x3");
    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "docking_x2_x3", result.tmv_flowpipes, setting);

	return 0;
}
