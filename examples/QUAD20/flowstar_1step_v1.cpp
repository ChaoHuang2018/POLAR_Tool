#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

/* ******** Argument list ********

argv[1]: order
argv[2]: x0_min
argv[3]: x0_max
argv[4]: x1_min
argv[5]: x1_max
argv[6]: x2_min
argv[7]: x2_max
argv[8]: x3_min
argv[9]: x3_max
argv[10]: x4_min
argv[11]: x4_max
argv[12]: x5_min
argv[13]: x5_max
argv[14]: u0_min
argv[15]: u0_max
argv[16]: u1_min
argv[17]: u1_max
argv[18]: u2_min
argv[19]: u2_max
argv[20]: step
argv[21]: net name


*/

int main(int argc, char *argvs[])
{
	string argv[36];
	size_t pos = 0;
	std::string s = argvs[1];
	//cout << "arguments: " << s << endl;
	std::string delimiter = "::";
	
	int i = 1;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		argv[i++] = s.substr(0, pos).c_str();
		s.erase(0, pos + delimiter.length());
		if(i == 36) break;
	}
	
	string benchmark_name = argv[2];

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
	Expression<Real> deriv_x0("0.25*(u0 + x1*x2)", vars); // theta_r = 0
	Expression<Real> deriv_x1("0.5*(u1 - 3*x0*x2)", vars);
	Expression<Real> deriv_x2("u2 + 2*x0*x1", vars);
	Expression<Real> deriv_x3("0.5*x1*(x3^2 + x4^2 + x5^2 - x5) + 0.5*x2*(x3^2 + x4^2 + x4 + x5^2) + 0.5*x0*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_x4("0.5*x0*(x3^2 + x4^2 + x5^2 + x5) + 0.5*x2*(x3^2 - x3 + x4^2 + x5^2) + 0.5*x1*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_x5("0.5*x0*(x3^2 + x4^2 - x4 + x5^2) + 0.5*x1*(x3^2 + x3 + x4^2 + x5^2) + 0.5*x2*(x3^2 + x4^2 + x5^2 + 1)", vars);
	Expression<Real> deriv_u0("0", vars);
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);

	vector<Expression<Real>> ode_rhs(numVars);
	ode_rhs[x0_id] = deriv_x0;
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[u0_id] = deriv_u0;
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);
	*/
	// Define the continuous dynamics.
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

	unsigned int order = stoi(argv[1]);

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
	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	// cout << argv[3] << argv[33] << endl;
	// double w = stod(argv[1]);
	int steps = 1;
	Interval 
			init_x1(stod(argv[4]), stod(argv[5])), 
			init_x2(stod(argv[6]), stod(argv[7])), 
			init_x3(stod(argv[8]), stod(argv[9])), 
			init_x4(stod(argv[10]), stod(argv[11])), 
			init_x5(stod(argv[12]), stod(argv[13])), 
			init_x6(stod(argv[14]), stod(argv[15])), 
			init_x7(stod(argv[16]), stod(argv[17])), 
			init_x8(stod(argv[18]), stod(argv[19])), 
			init_x9(stod(argv[20]), stod(argv[21])), // w=0.05
			init_x10(stod(argv[22]), stod(argv[23])), 
			init_x11(stod(argv[24]), stod(argv[25])), 
			init_x12(stod(argv[26]), stod(argv[27])),
			init_time(stod(argv[28]), stod(argv[29])),
			init_u0(stod(argv[30]), stod(argv[31])), 
			init_u1(stod(argv[32]), stod(argv[33])), 
			init_u2(stod(argv[34]), stod(argv[35]));
	
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
	X0.push_back(init_time);
	X0.push_back(init_u0);
	X0.push_back(init_u1);
	X0.push_back(init_u2);


	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	// no unsafe set
	vector<Constraint> safeSet;
	//vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// Always using symbolic remainder
	dynamics.reach(result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
	//dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);

	if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
	{
		initial_set = result.fp_end_of_time;
	}
	else
	{
		printf("Terminated due to too large overestimation.\n");
		exit(1);
	}

	cout.precision(17);
	Interval box;
	initial_set.tmvPre.tms[x1_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x2_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x3_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x4_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x5_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[x6_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;

	initial_set.tmvPre.tms[x7_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x8_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x9_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x10_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
    initial_set.tmvPre.tms[x11_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[x12_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;
	initial_set.tmvPre.tms[t_id].intEval(box, initial_set.domain);
	cout << scientific << box.inf() << " " << scientific << box.sup() << endl;




	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);

    const string dir_name = "./outputs/"  + benchmark_name + "_crown_flowstar";
    char* c = const_cast<char*>(dir_name.c_str());

	int mkres = mkdir(c, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
    plot_setting.setOutputDims("t", "x3");
    //plot_setting.plot_2D_octagon_MATLAB(c, "/" + to_string(stoi(argv[3])), result);
	plot_setting.plot_2D_octagon_MATLAB("./outputs/", "step_"  + to_string(stoi(argv[3])) + "_"  + to_string(1), result.tmv_flowpipes, setting);
    //plot_setting.plot_2D_octagon_MATLAB(c, "/x0_x3_" + to_string(stoi(argv[20])), result);
    // plot_setting.setOutputDims("x2", "x3");
	// plot_setting.plot_2D_octagon_MATLAB(c, "/x2_x3_" + to_string(stoi(argv[20])), result.tmv_flowpipes, setting);
    // //plot_setting.plot_2D_octagon_MATLAB(c, "/x1_x4_" + to_string(stoi(argv[20])), result);
    // plot_setting.setOutputDims("x4", "x5");
	// plot_setting.plot_2D_octagon_MATLAB(c, "/x4_x5_" + to_string(stoi(argv[20])), result.tmv_flowpipes, setting);
    //plot_setting.plot_2D_octagon_MATLAB(c, "/x2_x5_" + to_string(stoi(argv[20])), result);



	return 0;
}

