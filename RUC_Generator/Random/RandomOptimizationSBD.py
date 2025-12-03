def RandomOptimizationSBD(mask, rve_data, input_space, opt_settings, callback = None):
    """
    Optimize a random periodic microstructure given a segmented image.

    Arguments:
        mask            array       2D mask array of the fiber and matrix labels
        rve_data        dict        Dictionary containing RVE data (W, H, Nfibers)
        input_space     dict        Dictionary containing the input space the user
                                    wishes to optimize over
        opt_settings    dict        User defined Bayesian optimization settings
        nbins           int         number of bins to use for pdf calculations
        callback        func        callback function to update the streamlit app (optional)
    Outputs:

    """

    # Import Modules
    from baybe import Campaign
    from baybe.recommenders import (
        TwoPhaseMetaRecommender,
        NaiveHybridSpaceRecommender,
        )
    from baybe.parameters import (
        NumericalContinuousParameter,
        )
    from baybe.searchspace import SearchSpace
    from baybe.targets import NumericalTarget
    from baybe.objectives import SingleTargetObjective
    import numpy as np
    import pandas as pd
    import random
    from RUC_Generator.Random.RandomCharacterization import RandomCharacterization
    from RUC_Generator.Random.RandomSBD import RandomSBD

    if callback:
        callback('Initializing characterization')

    # Characterize the true microstructure
    VF = np.sum(mask == 1) / (len(mask) * len(mask[0]))
    VI = np.sum(mask == 3) / (len(mask) * len(mask[0]))
    _, _, _, pdf, _, _, _ = RandomCharacterization(mask, opt_settings['nbins'])

    # Define the input space
    # -- Defualt Values
    seed = 42
    damping = 0.5
    k = 1000
    gamma  = 0.1
    dt = 0.01
    max_steps = 10000
    mass = 1.0

    # -- Define possible input choices
    param_opts = ['seed', 'damping', 'k', 'gamma', 'dt']

    # -- Definte input parameters
    params = []
    for key in input_space.keys():
        params.append(
        NumericalContinuousParameter(
            name=key,         
            bounds=(input_space[key][0], input_space[key][1])       
            )
        )

    # Create the Campaign
    space = SearchSpace.from_product(parameters = params)
    obj = SingleTargetObjective(target=NumericalTarget(name='Target',minimize = True,))
    hybrid_recommender = TwoPhaseMetaRecommender(recommender=NaiveHybridSpaceRecommender())
    camp = Campaign(
        searchspace=space,
        objective=obj,
        recommender=hybrid_recommender,
        )
    
    # Create Initial Data
    df = pd.DataFrame()
    params_init = {}
    for i in range(opt_settings['batch size']):
        for j in range(len(params)):
            if i == 0:
                params_init[params[j].name] = []
            try:
                params_init[params[j].name].append(random.uniform(params[j].bounds.lower,params[j].bounds.upper))
            except:
                params_init[params[j].name].append(random.choice(list(params[j].values)))
    for key in params_init.keys():
        df[key] = params_init[key]

    # Evaluate the objective function
    q = []
    for i in df.index:
        try:
            if callback:
                callback(f"Initializing Microstructure {i+1}/{opt_settings['batch size']}")

            # Get Inputs
            input_array = {}
            for key in param_opts:
                if key in input_space.keys():
                    input_array[key] = df[key][i]
                else:
                    input_array[key] = eval(key)
            
            # Generate the mask
            if VI == 0:
                data = RandomSBD(
                                W=int(rve_data['W']), 
                                H=int(rve_data['H']), 
                                N_fibers=int(rve_data['Nfibers']), 
                                VF=float(VF), 
                                damping = float(input_array['damping']), 
                                k = float(input_array['k']), 
                                mass = mass,
                                gamma = float(input_array['gamma']),
                                dt = float(input_array['dt']), 
                                steps = max_steps, 
                                min_gap = int(rve_data['min_gap']), 
                                n_gen = 1, 
                                periodic = True, 
                                seed = float(input_array['seed'])
                                )
            else:
                data =  RandomSBD(
                                W=int(rve_data['W']), 
                                H=int(rve_data['H']), 
                                N_fibers=int(rve_data['Nfibers']), 
                                VF=float(VF), 
                                damping = float(input_array['damping']), 
                                k = float(input_array['k']), 
                                mass = mass,
                                gamma = float(input_array['gamma']),
                                dt = float(input_array['dt']), 
                                steps = max_steps, 
                                min_gap = int(rve_data['min_gap']), 
                                n_gen = 1, 
                                periodic = True, 
                                seed = float(input_array['seed']),
                                VI = VI
                                )
            mask_i = data[0][1]
            out_i = data[0][2]

            # Characterize the mask
            _, _, _, pdf_i, _, _, _ = RandomCharacterization(mask_i, nbins=opt_settings['nbins'])

            # Calculate mean squared error
            error = np.sqrt(np.sum((pdf_i - pdf)**2)) + out_i['Overlap']

            # Update measurement
            q.append(error)

        except Exception as e:
            if callback:
                callback(f"Exception at iteration {i}: {type(e).__name__} - {e}")
            q.append(100)

    # Add results to campaign
    df['Target'] = q
    camp.add_measurements(df)

    # Initialize Bayesian Optimization
    best_ovr = []
    best_batch = []
    patience_ct = 0
    current_batch = camp.measurements['BatchNr'].max()
    batch_df = camp.measurements[camp.measurements['BatchNr'] == current_batch]
    best_batch.append(batch_df['Target'][batch_df['Target'].idxmin()])
    best_ovr.append(camp.measurements['Target'][camp.measurements['Target'].idxmin()])
    best_prev = best_ovr[-1]

    # Run Bayesian Optimization
    for i in range(opt_settings['steps']):
        try:
            if callback:
                callback(f"Bayesian Optimization Iteration {i+1}/{opt_settings['steps']}")

            # Create the data frame with BO recommendations
            df_bo = camp.recommend(batch_size=opt_settings['batch size'])

            # Evaluate recommendations
            q_bo = []
            for j in df_bo.index:
                if callback:
                    callback(   f"    Bayesian Recommendation {j+1}/{opt_settings['batch size']}")

                # Get Inputs
                input_array = {}
                for key in param_opts:
                    if key in input_space.keys():
                        input_array[key] = df_bo[key][j]
                    else:
                        input_array[key] = eval(key)
                
                # Generate the mask
                if VI == 0:
                    data = RandomSBD(
                                W=int(rve_data['W']), 
                                H=int(rve_data['H']), 
                                N_fibers=int(rve_data['Nfibers']), 
                                VF=float(VF), 
                                damping = float(input_array['damping']), 
                                k = float(input_array['k']), 
                                mass = mass,
                                gamma = float(input_array['gamma']),
                                dt = float(input_array['dt']), 
                                steps = max_steps, 
                                min_gap = int(rve_data['min_gap']), 
                                n_gen = 1, 
                                periodic = True, 
                                seed = float(input_array['seed'])
                                )
                else:
                    data = RandomSBD(
                                W=int(rve_data['W']), 
                                H=int(rve_data['H']), 
                                N_fibers=int(rve_data['Nfibers']), 
                                VF=float(VF), 
                                damping = float(input_array['damping']), 
                                k = float(input_array['k']), 
                                mass = mass,
                                gamma = float(input_array['gamma']),
                                dt = float(input_array['dt']), 
                                steps = max_steps, 
                                min_gap = int(rve_data['min_gap']), 
                                n_gen = 1, 
                                periodic = True, 
                                seed = float(input_array['seed']),
                                VI = VI
                                )
                    
                mask_j = data[0][1]
                out_j = data[0][2]

                # Characterize the mask
                _, _, _, pdf_j, _, _, _ = RandomCharacterization(mask_j, nbins=opt_settings['nbins'])
                
                # Calculate mean squared error
                error = np.sqrt(np.sum((pdf_j - pdf)**2)) + out_j['Overlap']

                # Update measurement
                q_bo.append(error)
        except Exception as e:
            if callback:
                callback(f"Exception at iteration {i}: {e}")
            q.append(100)

        # Add simulations to campaign
        df_bo['Target']=q_bo
        camp.add_measurements(df_bo)

        # Get Information
        current_batch = camp.measurements['BatchNr'].max()
        batch_df = camp.measurements[camp.measurements['BatchNr'] == current_batch]
        best_batch.append(batch_df['Target'][batch_df['Target'].idxmin()])
        best_ovr.append(camp.measurements['Target'][camp.measurements['Target'].idxmin()])

        # Check patience
        if best_ovr[-1] == best_prev:
            patience_ct = patience_ct + 1
            if patience_ct == opt_settings['patience']:
                break
        else:
            patience_ct = 1
            best_prev = best_ovr[-1]

        if callback:
            callback(f"Best Batch Error = {best_batch[-1]}")
            callback(f"Best Current Error = {best_ovr[-1]}")

    # Get best microstructure
    best_idx = camp.measurements['Target'].idxmin()
    input_array = {}
    for key in param_opts:
        if key in input_space.keys():
            input_array[key] = camp.measurements[key][best_idx]
        else:
            input_array[key] = eval(key)
    
    # Generate the mask
    if VI == 0:
        data = RandomSBD(
                    W=int(rve_data['W']), 
                    H=int(rve_data['H']), 
                    N_fibers=int(rve_data['Nfibers']), 
                    VF=float(VF), 
                    damping = float(input_array['damping']), 
                    k = float(input_array['k']), 
                    mass = mass,
                    gamma = float(input_array['gamma']),
                    dt = float(input_array['dt']), 
                    steps = max_steps, 
                    min_gap = int(rve_data['min_gap']), 
                    n_gen = 1, 
                    periodic = True, 
                    seed = float(input_array['seed'])
                    )
    else:
        data = RandomSBD(
                    W=int(rve_data['W']), 
                    H=int(rve_data['H']), 
                    N_fibers=int(rve_data['Nfibers']), 
                    VF=float(VF), 
                    damping = float(input_array['damping']), 
                    k = float(input_array['k']), 
                    mass = mass,
                    gamma = float(input_array['gamma']),
                    dt = float(input_array['dt']), 
                    steps = max_steps, 
                    min_gap = int(rve_data['min_gap']), 
                    n_gen = 1, 
                    periodic = True, 
                    seed = float(input_array['seed']),
                    VI=VI
                    )
    best_mask = data[0][1]
    best_out = data[0][2]
    _, _, _, pdf_best, _, _, _ = RandomCharacterization(best_mask, nbins=opt_settings['nbins'])
    best_error = np.sqrt(np.sum((pdf_best - pdf)**2)) + best_out['Overlap']

    return best_mask, best_out, best_error