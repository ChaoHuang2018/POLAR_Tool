#include "../../POLAR/NeuralNetwork.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{

    string comb = "Mix";

 //   intervalNumPrecision = 300;

	// Declaration of the state variables.
	unsigned int numVars = 19;

    Variables vars;

	// intput format (\omega_1, \psi_1, \omega_2, \psi_2, \omega_3, \psi_3)
	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");
	int x7_id = vars.declareVar("x7");
	int x8_id = vars.declareVar("x8");
	int x9_id = vars.declareVar("x9");
	int x10_id = vars.declareVar("x10");
	int x11_id = vars.declareVar("x11");
	int x12_id = vars.declareVar("x12");
    int t_id = vars.declareVar("t");
	
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");
    int u3_id = vars.declareVar("u3");
    int u4_id = vars.declareVar("u4");
    int u5_id = vars.declareVar("u5");
    int u6_id = vars.declareVar("u6");

	int domainDim = numVars + 1;

    ODE<Real> dynamics({"x6*(sin(x7)*sin(x9) + cos(x7)*cos(x9)*sin(x8)) - x5*(cos(x7)*sin(x9) - cos(x9)*sin(x7)*sin(x8)) + x4*cos(x8)*cos(x9)",
    	"x5*(cos(x7)*cos(x9) + sin(x7)*sin(x8)*sin(x9)) - x6*(cos(x9)*sin(x7) - cos(x7)*sin(x8)*sin(x9)) + x4*cos(x8)*sin(x9)",
		"x6*cos(x7)*cos(x8) - x4*sin(x8) + x5*cos(x8)*sin(x7)", "u1 - sin(x8) + x5*x10 - x6*x12",
		"u2 + cos(x8)*sin(x7) - x4*x10 + x6*x11",
		"u3 + cos(x7)*cos(x8) + x4*x12 - x5*x11",
		"x11 + (x10*cos(x7)*sin(x8))/cos(x8) + (x12*sin(x7)*sin(x8))/cos(x8)",
		"x12*cos(x7) - x10*sin(x7)",
		"(x10*cos(x7))/cos(x8) + (x12*sin(x7))/cos(x8)", "u6", "u4", "u5", "1"}, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.05, order);

	// time horizon for a single control step
//	setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-7);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

//	setting.prepare();
    //setting.g_setting.prepareForReachability(15);
  //  cout << "--------" << setting.g_setting.factorial_rec.size() << ", " << setting.g_setting.power_4.size() << ", " << setting.g_setting.double_factorial.size() << endl;

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	
	int steps = 10;
	Interval
        init_x1(0),
        init_x2(0),
        init_x3(0),
        init_x4(1, 1),
        init_x5(1, 1),
        init_x6(1, 1),
        init_x7(1, 1),
        init_x8(1, 1),
        init_x9(1, 1),
        init_x10(0),
        init_x11(0),
        init_x12(0),
        init_t(0);


	Interval init_u1(0), init_u2(0), init_u3(0), init_u4(0), init_u5(0), init_u6(0);
	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_x7);
	X0.push_back(init_x8);
	X0.push_back(init_x9);
	X0.push_back(init_x10);
	X0.push_back(init_x11);
	X0.push_back(init_x12);
	X0.push_back(init_t);
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);
    X0.push_back(init_u4);
    X0.push_back(init_u5);
    X0.push_back(init_u6);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

    Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet = {Constraint("-x1 - 0.5", vars), Constraint("x1 - 0.5", vars),
			Constraint("-x6 - 1", vars), Constraint("x6 - 1", vars),
			Constraint("-x7 - 1", vars), Constraint("x7 - 1", vars),
			Constraint("-x8 - 1", vars), Constraint("x8 - 1", vars)
			};

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "controller_airplane_POLAR";
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	unsigned int bernstein_order = 2;
	unsigned int partition_num = 100;

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

		for (int i = 0; i < 12; i++)
		{
			tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
		}


		// taylor propagation
        // taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, comb, "Concrete");
        TaylorModelVec<Real> tmv_output;


        if (if_symbo == 0)
        {
            // not using symbolic remainder
            nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
        }
        else
        {
            // using symbolic remainder
            nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
        }

        // tmv_output.output(cout, vars);
//		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
//		tmv_output.Remainder(rm1);
//		cout << "Neural network taylor remainder: " << rm1 << endl;


        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2];
        initial_set.tmvPre.tms[u4_id] = tmv_output.tms[3];
        initial_set.tmvPre.tms[u5_id] = tmv_output.tms[4];
        initial_set.tmvPre.tms[u6_id] = tmv_output.tms[5];
//        cout << "TM -- Propagation" << endl;


        // dynamics.reach(result, setting, initial_set, unsafeSet);
        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);

        if(result.status == COMPLETED_UNSAFE)
        {
        	break;
        }

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			break;
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


	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("x1", "x6");
    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "airplane_x1_x6", result.tmv_flowpipes, setting);
    
    plot_setting.setOutputDims("x1", "x4");
    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "airplane_x1_x4", result.tmv_flowpipes, setting);
    
//    plot_setting.setOutputDims("t", "x9");
//    plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "airplane_", result.tmv_flowpipes, setting);

	return 0;
}
