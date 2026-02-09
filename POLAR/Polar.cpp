#include "Polar.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <chrono>
#include <iomanip>

using namespace flowstar;
using namespace std;

void nncs_reachability(System s, Specification spec, PolarSetting ps)
{
    NeuralNetwork nn = s.nn;
    unsigned int numVars = s.num_of_states + s.num_of_control + 1;
    
    intervalNumPrecision = ps.get_interval_precision();
    
    Variables vars;
    vector<int> var_id_list;
    for (int i = 0; i < s.num_of_states; i++)
    {
        int temp_var_id = vars.declareVar(s.state_name_list[i]);
        var_id_list.push_back(temp_var_id);
    }
    int t_id = vars.declareVar("t");
    var_id_list.push_back(t_id);
    for (int i = 0; i < s.num_of_control; i++)
    {
        int temp_var_id = vars.declareVar(s.control_name_list[i]);
        var_id_list.push_back(temp_var_id);
    }
    int domainDim = numVars + 1;
 /*
    vector<Expression<Real>> ode_rhs(numVars);
    for (int i = 0; i < s.num_of_states; i++)
    {
        Expression<Real> temp_deriv(s.ode_list[i], vars);
        ode_rhs[var_id_list[i]] = temp_deriv;
    }
    for (int i = s.num_of_states; i < s.num_of_states+s.num_of_control; i++)
    {
        Expression<Real> temp_deriv("0", vars);
        ode_rhs[var_id_list[i]] = temp_deriv;
    }
*/
//    cout << "1" << endl;

    vector<string> ode_list = s.ode_list;
    ode_list.push_back("1");
    for (int i = 0; i < s.num_of_control; i++)
    {
        ode_list.push_back("0");
    }
    ODE<Real> dynamics(ode_list, vars);
    
    
    // Flow* setting
    Computational_Setting setting(vars);
    
    // stepsize and order for reachability analysis
    setting.setFixedStepsize(ps.get_flowpipe_stepsize(), ps.get_taylor_order());
//    cout << "taylor order: " << ps.get_taylor_order() << endl;

    setting.printOff();

    // cutoff threshold
    setting.setCutoffThreshold(ps.get_cutoff_threshold());

    // remainder estimation
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(numVars, I);
    setting.setRemainderEstimation(remainder_estimation);


    int steps = spec.time_steps;
    
//    cout << "2" << endl;
    vector<Interval> init;
    // state init
    init = spec.init;
    // t init
    init.push_back(Interval(0));
    // control input init
    for (int i = 0; i < s.num_of_control; i++)
    {
        init.push_back(Interval(0));
    }
    Flowpipe initial_set(init);
    
    Symbolic_Remainder symbolic_remainder(initial_set, ps.get_symbolic_queue_size());
    
    // no unsafe set
//    vector<Constraint> safeSet;
    // safe set
    vector<Constraint> safeSet;
//    cout << "123:" << spec.safe_set.size() << endl;
    for (int i = 0; i < spec.safe_set.size(); i++)
    {
//        cout << "111" << endl;
//        cout << spec.safe_set[i] << endl;
        Constraint cons_temp(spec.safe_set[i], vars);
//        cout << "222" << endl;
        // safeSet.push_back(cons_temp);
    }
//    cout << "3" << endl;
    // result of the reachability computation
    Result_of_Reachability result;

    Interval cutoff_threshold(-1e-7, 1e-7);
    
    double err_max = 0;
    time_t start_timer;
    time_t end_timer;
    double seconds;
    time(&start_timer);

    std::vector<std::vector<double>> remainder_widths;  // [step][state_dim]
    remainder_widths.reserve(steps);
    const double fail_fill = (ps.has_fail_fill_width() ? ps.get_fail_fill_width() : 1e12);
    int fail_step = -1;

    
    for (int iter = 0; iter < steps; ++iter)
    {
        cout << "Step " << iter << " starts.      " << endl;
        
        TaylorModelVec<Real> tmv_input;
        for (int i = 0; i < s.num_of_states; i++)
        {
            tmv_input.tms.push_back(initial_set.tmvPre.tms[i]);
//            Interval I(-0.1,0.1);
//            tmv_input.tms[i].remainder += I;
            
//            initial_set.tmvPre.tms[i].output(cout, vars);
//            cout << endl;
        }
        
        TaylorModelVec<Real> tmv_output;
        if (ps.get_remainder_type() == 0) //"Concrete")
        {
            // not using symbolic remainder
            nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, ps, setting);
        }
        else
        {
            // using symbolic remainder
            // cout << "1" << endl;
            nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, ps, setting);
            // cout << "2" << endl;
        }
        
       Matrix<Interval> rm1(1, 1);
       tmv_output.Remainder(rm1);
       cout << "Neural network taylor remainder: " << rm1 << endl;
    //    cout << tmv_output.tms[0].remainder << endl;

        Matrix<Real> coefficients(tmv_output.tms.size(), 2);
		tmv_output.tms[0].expansion.linearCoefficients(coefficients, 0);
		cout << "Linear coefficient: " << endl;
		for(int i = 0; i < 2; i++) {
			cout << coefficients[0][i] << endl;
		}

        
        for (int i = 0; i < s.num_of_control; i++)
        {
            initial_set.tmvPre.tms[var_id_list[s.num_of_states + 1 + i]] = tmv_output.tms[i];
//            initial_set.tmvPre.tms[var_id_list[s.num_of_states + i]].output(cout, vars);
//            cout << endl;
        }
        
//        cout << "size: " << initial_set.tmvPre.tms.size() << endl;
        
        dynamics.reach(result, initial_set, s.control_stepsize, setting, safeSet, symbolic_remainder);
//      dynamics.reach(result, initial_set, s.control_stepsize, setting, safeSet);

        
        if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
            std::vector<double> widths_this_step;
            widths_this_step.reserve(s.num_of_states);

            for (int vi = 0; vi < s.num_of_states; ++vi) {
                try {
                    widths_this_step.push_back(initial_set.tmv.tms[vi].remainder.width());
                } catch (...) {
                    widths_this_step.push_back(std::numeric_limits<double>::quiet_NaN());  // 容错
                }
            }
            remainder_widths.push_back(std::move(widths_this_step));
		}
		else
		{
			// printf("Terminated due to too large overestimation.\n");
            // std::cout << "Terminated early: remainder exploded at step " << iter << ".\n";
            fail_step = iter;

            std::vector<double> fill_vec(s.num_of_states, fail_fill);

            // 当前步以及剩余步都填充同一个大值（保证向量长度 == steps）
            for (int k = iter; k < steps; ++k) {
                remainder_widths.push_back(fill_vec);
            }
            break;
		}
    }
    
//    vector<Interval> end_box;
//    string reach_result;
//    reach_result = result.status;
//    result.fp_end_of_time.intEval(end_box, ps.get_taylor_order(), setting.tm_setting.cutoff_threshold);
//    cout << "4" << endl;
    time(&end_timer);
    seconds = difftime(start_timer, end_timer);
    
    result.transformToTaylorModels(setting);

    // === write json===
    try {
        // ./outputs/<output_filename>.json
        std::string out_dir = "./outputs/";
        int mkres2 = mkdir(out_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        (void)mkres2; 

        std::string json_path = out_dir + ps.get_output_filename() + ".json";

        // time
        auto now = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm tm_utc;
    #ifdef _WIN32
        gmtime_s(&tm_utc, &tt);
    #else
        gmtime_r(&tt, &tm_utc);
    #endif
        std::ostringstream ts;
        ts << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");

        std::cerr << "[POLAR-DBG] remainder_widths shape: steps=" 
          << remainder_widths.size()
          << " x dims=" << (remainder_widths.empty()?0:remainder_widths[0].size())
          << std::endl;

        // a record
        nlohmann::json rec;
        rec["timestamp"] = ts.str();
        rec["output_filename"] = ps.get_output_filename();
        rec["step_size"] = s.control_stepsize;
        rec["taylor_order"] = ps.get_taylor_order();
        rec["num_steps"] = (int)remainder_widths.size();
        rec["remainder_widths"] = remainder_widths;   // 数组：
        // 
        rec["state_dim"] = s.num_of_states;
        rec["control_dim"] = s.num_of_control;

        // append
        std::ofstream ofs(json_path, std::ios::out | std::ios::app);
        if (ofs.is_open()) {
            ofs << rec.dump() << '\n';
            ofs.close();
            std::cout << "[POLAR] Appended widths to " << json_path << std::endl;
        } else {
            std::cerr << "[POLAR] Failed to open " << json_path << " for append.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "[POLAR] Failed to append widths JSON: " << e.what() << std::endl;
    }

    if (ps.if_plot == true) {
        // plot the flowpipes in the x-y plane
        Plot_Setting plot_setting(vars);
        plot_setting.setOutputDims(ps.get_output_dim()[0], ps.get_output_dim()[1]);

        int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        if (mkres < 0 && errno != EEXIST)
        {
            printf("Can not create the directory for images.\n");
            exit(1);
        }

        // std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";
        
        // cout << running_time << endl;
        
        plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", ps.get_output_filename(), result.tmv_flowpipes, setting);
    }
}
