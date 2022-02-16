#include "../flowstar-toolbox/Continuous.h"

using namespace flowstar;
using namespace std;


int main()
{
	unsigned int numVars = 8;

	int x1_id = stateVars.declareVar("x1");
	int x2_id = stateVars.declareVar("x2");
	int x3_id = stateVars.declareVar("x3");
	int x4_id = stateVars.declareVar("x4");
	int x5_id = stateVars.declareVar("x5");
	int x6_id = stateVars.declareVar("x6");
	int x7_id = stateVars.declareVar("x7");
	int t_id = stateVars.declareVar("t");



	// define the dynamics
	Expression_AST<Real> ode_expression_x1("1.4 * x3 - 0.9 * x1");
	Expression_AST<Real> ode_expression_x2("2.5 * x5 - 1.5 * x2");
	Expression_AST<Real> ode_expression_x3("0.6 * x7 - 0.8 * x3 * x2");
	Expression_AST<Real> ode_expression_x4("2 - 1.3 * x4 * x3");
	Expression_AST<Real> ode_expression_x5("0.7 * x1 - x4 * x5");
	Expression_AST<Real> ode_expression_x6("0.3 * x1 - 3.1 * x6");
	Expression_AST<Real> ode_expression_x7("1.8 * x6 - 1.5 * x7 * x2");
	Expression_AST<Real> ode_expression_t("1");



	vector<Expression_AST<Real> > ode_rhs(numVars);
	ode_rhs[x1_id] = ode_expression_x1;
	ode_rhs[x2_id] = ode_expression_x2;
	ode_rhs[x3_id] = ode_expression_x3;
	ode_rhs[x4_id] = ode_expression_x4;
	ode_rhs[x5_id] = ode_expression_x5;
	ode_rhs[x6_id] = ode_expression_x6;
	ode_rhs[x7_id] = ode_expression_x7;
	ode_rhs[t_id] = ode_expression_t;



	Deterministic_Continuous_Dynamics dynamics(ode_rhs);



	// set the reachability parameters
	Computational_Setting setting;

	// set the stepsize and the order
//	setting.setFixedStepsize(0.02, 3, 8);			// adaptive orders
	setting.setAdaptiveStepsize(0.01, 0.2, 4);			// adaptive stepsize

	// set the time horizon
	setting.setTime(20);

	// set the cutoff threshold
	setting.setCutoffThreshold(1e-6);

	// print out the computation steps
	setting.printOn();

	// set up the remainder estimation
	Interval I(-1e-3, 1e-3);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	// call this function when all of the parameters are defined
	setting.prepare();

	double w = 0.1;

	// define the initial set which is a box
	Interval init_x1(1.2-w, 1.2+w), init_x2(1.05-w, 1.05+w), init_x3(1.5-w, 1.5+w), init_x4(2.4-w, 2.4+w),
			init_x5(1-w, 1+w), init_x6(0.1-w, 0.1+w), init_x7(0.45-w, 0.45+w), init_t;

	vector<Interval> initial_box(numVars);
	initial_box[x1_id] = init_x1;
	initial_box[x2_id] = init_x2;
	initial_box[x3_id] = init_x3;
	initial_box[x4_id] = init_x4;
	initial_box[x5_id] = init_x5;
	initial_box[x6_id] = init_x6;
	initial_box[x7_id] = init_x7;
	initial_box[t_id] = init_t;


	Flowpipe initialSet(initial_box);

	// defining the symbolic remainder, the first argument is the initial set which should be a flowpipe,
	// and the second argument is the original queue size
	Symbolic_Remainder symbolic_remainder(initialSet, 500);

	// the unsafe set
	vector<Constraint> unsafeSet;
	Constraint constraint("5 - x4");
	unsafeSet.push_back(constraint);


	/*
	 * The structure of the class Result_of_Reachability is defined as below:
	 * nonlinear_flowpipes: the list of computed flowpipes
	 * tmv_flowpipes: translation of the flowpipes, they will be used for further analysis
	 * fp_end_of_time: the flowpipe at the time T
	 */
	Result_of_Reachability result;

	// run the reachability computation
	clock_t begin, end;
	begin = clock();

	// with symbolic remainder
	// the object symbolic_remainder can be reused by another continuous dynamics if there is no reset
	dynamics.reach_sr(result, setting, initialSet, symbolic_remainder, unsafeSet);

	// without symbolic remainder
//	dynamics.reach(result, setting, initialSet, unsafeSet);

	end = clock();
	printf("time cost: %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);


	switch(result.status)
	{
	case COMPLETED_SAFE:
		printf("Safe\n");
		break;
	case COMPLETED_UNSAFE:
		printf("Unsafe\n");
		break;
	case COMPLETED_UNKNOWN:
		printf("Unknown\n");
		break;
	default:
		printf("Fail to compute flowpipes.\n");
	}


	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting;
	plot_setting.printOn();
	plot_setting.setOutputDims(t_id, x4_id);

	plot_setting.plot_2D_interval_GNUPLOT("./outputs/", "laubloomis", result);

	return 0;
}

