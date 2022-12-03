#include "Polar.h"

using namespace flowstar;
using namespace std;

void nncs_reachability(System s, Specification spec, PolarSetting ps)
{
    NeuralNetwork nn = s.nn;
    unsigned int numVars = s.num_of_states + s.num_of_control;
    
    intervalNumPrecision = 300;
    
    Variables vars;
    vector<int> var_id_list;
    for (int i = 0; i < s.num_of_states; i++)
    {
        int temp_var_id = vars.declareVar(s.state_name_list[i]);
        var_id_list.push_back(temp_var_id);
    }
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
    ODE<Real> dynamics(s.ode_list, vars);
    
    
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
    init = spec.init;
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
        safeSet.push_back(cons_temp);
    }
//    cout << "3" << endl;
    // result of the reachability computation
    Result_of_Reachability result;
    
    double err_max = 0;
    time_t start_timer;
    time_t end_timer;
    double seconds;
    time(&start_timer);
    
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
            cout << "1" << endl;
            nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, ps, setting);
            cout << "2" << endl;
        }
        
//        Matrix<Interval> rm1(1, 1);
//        tmv_output.Remainder(rm1);
//        cout << "Neural network taylor remainder: " << rm1 << endl;
//        cout << tmv_output.tms[0].remainder << endl;
        
        for (int i = 0; i < s.num_of_control; i++)
        {
            initial_set.tmvPre.tms[var_id_list[s.num_of_states + i]] = tmv_output.tms[i];
//            initial_set.tmvPre.tms[var_id_list[s.num_of_states + i]].output(cout, vars);
//            cout << endl;
        }
        
//        cout << "size: " << initial_set.tmvPre.tms.size() << endl;
        
        dynamics.reach(result, initial_set, s.control_stepsize, setting, safeSet, symbolic_remainder);
//      dynamics.reach(result, initial_set, s.control_stepsize, setting, safeSet);

        
        if (result.isCompleted())
        {
            initial_set = result.fp_end_of_time;
//            cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
        }
        else
        {
            printf("Terminated due to too large overestimation.\n");
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

    // plot the flowpipes in the x-y plane
    Plot_Setting plot_setting(vars);
    plot_setting.setOutputDims(ps.get_output_dim()[0], ps.get_output_dim()[1]);

    int mkres = mkdir("../outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
    if (mkres < 0 && errno != EEXIST)
    {
        printf("Can not create the directory for images.\n");
        exit(1);
    }

    std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";
    
    cout << running_time << endl;
    
    plot_setting.plot_2D_octagon_GNUPLOT("../outputs/", ps.get_output_filename(), result.tmv_flowpipes, setting);
}
