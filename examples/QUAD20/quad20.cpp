#include "../../POLAR/NeuralNetwork.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{

    string comb = argv[6];

    intervalNumPrecision = 300;

	// Declaration of the state variables.
	unsigned int numVars = 16;

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
	int u0_id = vars.declareVar("u0");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");

	int domainDim = numVars + 1;

	/*
	// Define the continuous dynamics.
	Expression<Real> deriv_x1("cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6", vars); // theta_r = 0
	Expression<Real> deriv_x2("cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6", vars);
	Expression<Real> deriv_x3("sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6", vars);
	Expression<Real> deriv_x4("x12*x5 - x11*x6 - 9.81*sin(x8)", vars);
	Expression<Real> deriv_x5("x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)", vars);
	Expression<Real> deriv_x6("x11*x4 - x10*x5 + 9.81*cos(x8)*cos(x7) - 9.81 - u0 / 1.4", vars);
	Expression<Real> deriv_x7("x10 + (sin(x7)*(sin(x8)/cos(x8)))*x11 + (cos(x7)*(sin(x8)/cos(x8)))*x12", vars);
	Expression<Real> deriv_x8("cos(x7)*x11 - sin(x7)*x12", vars);
	Expression<Real> deriv_x9("(sin(x7)/cos(x8))*x11 + (cos(x7)/cos(x8))*x12", vars);
	Expression<Real> deriv_x10("-0.92592592592593*x11*x12 + 18.51851851851852*u1", vars);
	Expression<Real> deriv_x11("0.92592592592593*x10*x12 + 18.51851851851852*u2", vars);
	Expression<Real> deriv_x12("0", vars);
    Expression<Real> deriv_t("1", vars);
	Expression<Real> deriv_u0("0", vars);
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	ode_rhs[x7_id] = deriv_x7;
	ode_rhs[x8_id] = deriv_x8;
	ode_rhs[x9_id] = deriv_x9;
	ode_rhs[x10_id] = deriv_x10;
	ode_rhs[x11_id] = deriv_x11;
	ode_rhs[x12_id] = deriv_x12;
    ode_rhs[t_id] = deriv_t;
	ode_rhs[u0_id] = deriv_u0;
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	ODE<Real> dynamics({"cos(x8)*cos(x9)*x4 + (sin(x7)*sin(x8)*cos(x9) - cos(x7)*sin(x9))*x5 + (cos(x7)*sin(x8)*cos(x9) + sin(x7)*sin(x9))*x6",
						"cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6",
						"sin(x8)*x4 - sin(x7)*cos(x8)*x5 - cos(x7)*cos(x8)*x6",
						"x12*x5 - x11*x6 - 9.81*sin(x8)",
						"x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)",
						"x11*x4 - x10*x5 + 9.81*cos(x8)*cos(x7) - 9.81 - u0 / 1.4",
						"x10 + (sin(x7)*(sin(x8)/cos(x8)))*x11 + (cos(x7)*(sin(x8)/cos(x8)))*x12",
						"cos(x7)*x11 - sin(x7)*x12",
						"(sin(x7)/cos(x8))*x11 + (cos(x7)/cos(x8))*x12",
						"-0.92592592592593*x11*x12 + 18.51851851851852*u1",
						"0.92592592592593*x10*x12 + 18.51851851851852*u2",
						"0",
						"1","0","0","0"}, vars);
	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.005, order);

	// time horizon for a single control step
	//setting.setTime(0.1);

	// cutoff threshold
	setting.setCutoffThreshold(1e-7);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	//setting.prepare();
    //setting.g_setting.prepareForReachability(15);
  //  cout << "--------" << setting.g_setting.factorial_rec.size() << ", " << setting.g_setting.power_4.size() << ", " << setting.g_setting.double_factorial.size() << endl;

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stod(argv[1]);
	int steps = stoi(argv[2]);
	Interval
        init_x1(0 - w, 0 + w),
        init_x2(0 - w, 0 + w),
        init_x3(0 - w, 0 + w),
        init_x4(0 - w, 0 + w),
        init_x5(0 - w, 0 + w),
        init_x6(0 - w, 0 + w),
        init_x7(0),
        init_x8(0),
        init_x9(0),
        init_x10(0),
        init_x11(0),
        init_x12(0),
        init_t(0);
	// Interval init_x0(-0.25 - w, -0.25 + w), init_x1(-0.25 - w, -0.25 + w), init_x2(0.35 - w, 0.35 + w), init_x3(-0.35 - w, -0.35 + w), init_x4(0.45 - w, 0.45 + w), init_x5(-0.35 - w, -0.35 + w);
	Interval init_u0(0), init_u1(0), init_u2(0);
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
	X0.push_back(init_u0);
	X0.push_back(init_u1);
	X0.push_back(init_u2);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

    Symbolic_Remainder symbolic_remainder(initial_set, 2000);

	// no unsafe set
	vector<Constraint> safeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "quad_controller_3_64";
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-7, 1e-7);
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;

	unsigned int if_symbo = stoi(argv[5]);

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
    time_t flowstar_start_timer;
    time_t nn_start_timer;
    time_t flowstar_end_timer;
    time_t nn_end_timer;
    double flowstar_seconds = 0;
    double nn_seconds = 0;
	time(&start_timer);

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}

    int run_step = 0;

	// perform 35 control steps
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

	    time(&nn_start_timer);

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

	    time(&nn_end_timer);
	    nn_seconds += difftime(nn_start_timer, nn_end_timer);

        // tmv_output.output(cout, vars);
		Matrix<Interval> rm1(nn.get_num_of_outputs(), 1);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;


        initial_set.tmvPre.tms[u0_id] = tmv_output.tms[0];
        initial_set.tmvPre.tms[u1_id] = tmv_output.tms[1];
        initial_set.tmvPre.tms[u2_id] = tmv_output.tms[2];
        cout << "TM -- Propagation" << endl;

        time(&flowstar_start_timer);
        // dynamics.reach(result, setting, initial_set, unsafeSet);
        dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
        time(&flowstar_end_timer);

        flowstar_seconds += difftime(flowstar_start_timer, flowstar_end_timer);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			break;
		}
        run_step = iter;
	}

	vector<Interval> end_box;
	string reach_result;
	reach_result = "Verification result: " + to_string(run_step);
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);

    TaylorModelVec<Real> tm_;
    result.fp_end_of_time.compose(tm_, order, setting.tm_setting.cutoff_threshold);
    Interval tmRange;
    tm_.tms[0].intEval(tmRange, result.fp_end_of_time.domain);

	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

    string benchmark_name = "quad_w" + to_string(int(stod(argv[1]) * 1000)) + "_order" + to_string(stoi(argv[4])) + "_" + comb;
	int mkres = mkdir("./outputs/", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
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
        result_output << "NN propagation time: " << -nn_seconds << endl;
        result_output << "flowstar time: " << -flowstar_seconds << endl;
        result_output << "Remainder range: " << tm_.tms[x3_id].remainder << endl;
        result_output << "Remainder size: " << tm_.tms[x3_id].remainder.sup() - tm_.tms[x3_id].remainder.inf() << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("t", "x3");
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", benchmark_name + "_" + "_"  + to_string(if_symbo), result.tmv_flowpipes, setting);
    //plot_setting.plot_2D_octagon_MATLAB("./outputs/" + benchmark_name, "_" + to_string(if_symbo), result);

	return 0;
}
