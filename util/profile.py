import pstats

def show_func_runtimes(file):
    functions_of_interest = [
        "SCP_OF_SLS.py:76(solve)",
        "fast_OF_SLS.py:178(forward_solve)", 
        "fast_OF_SLS.py:294(evaluate_dual_eta)",
        "fast_OF_SLS.py:368(backward_solve)",
        "fast_OF_SLS.py:484(assemble_output_feedback_matrices)",
        "fast_OF_SLS.py:537(update_tightening)",
        "fast_OF_SLS.py:686(initialize_tube_cost_fun)",
        "SCP_OF_SLS.py:295(initialize_jacobian_Function)",
        "SCP_OF_SLS.py:335(initialize_tube_cost_quadratization)",
        "SCP_OF_SLS.py:145(solve_nominal_trajectory)",
        "SCP_OF_SLS.py:446(update_jacobian)",
        "SCP_OF_SLS.py:394(update_traj_cost_wrt_tube)",
        "qp.py:55(solve)",
        "qp.py:346(update_dynamics)"
    ]

    print(f"==== ==== Profile ==== ====")
    stats = pstats.Stats(file)
    stats.strip_dirs()

    entries = []
    append_entry = entries.append
    for func, stat in stats.stats.items():
        filename, line, funcname = func
        append_entry((f"{filename}:{line}({funcname})", stat[3]))

    for f in functions_of_interest:
        cumulative_time = 0.0
        for name, ct in entries:
            if f in name:
                cumulative_time += ct
        print(f"{f}: {cumulative_time:.6f}s")

def show_func_runtimes_SDP(file):
    functions_of_interest = [
        "SCP_OF_SLS_SDP.py:71(solve)",
        "fast_OF_SLS_SDP.py:177(forward_solve)", 
        "fast_OF_SLS_SDP.py:271(evaluate_dual_eta)",
        "fast_OF_SLS_SDP.py:344(backward_solve)",
        "fast_OF_SLS_SDP.py:389(update_tightening)",
        "fast_OF_SLS_SDP.py:522(initialize_tube_cost_fun)",
        "SCP_OF_SLS_SDP.py:194(initialize_jacobian_Function)",
        "SCP_OF_SLS_SDP.py:232(initialize_tube_cost_quadratization)",
        "SCP_OF_SLS_SDP.py:130(solve_nominal_trajectory)",
        "SCP_OF_SLS_SDP.py:344(update_jacobian)",
        "SCP_OF_SLS_SDP.py:293(update_traj_cost_wrt_tube)",
        "qp.py:55(solve)",
        "qp.py:346(update_dynamics)"
    ]

    print(f"==== ==== Profile ==== ====")
    stats = pstats.Stats(file)
    stats.strip_dirs()

    entries = []
    append_entry = entries.append
    for func, stat in stats.stats.items():
        filename, line, funcname = func
        append_entry((f"{filename}:{line}({funcname})", stat[3]))

    for f in functions_of_interest:
        cumulative_time = 0.0
        for name, ct in entries:
            if f in name:
                cumulative_time += ct
        print(f"{f}: {cumulative_time:.6f}s")
