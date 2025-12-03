def UI_Definitions(key, vars = None, tag = None):
    """
    Get the values for user inputs for the streamlit UI.

    Arguments:
        key     string      keyword for which values to grab
        vars    dict        streamlit session state
        tag     string      option for specific modules

    Outputs:
        depends on key
    """

    # Import Modules
    import math

    # Import Functions
    from RUC_Generator.Hexagonal.Hex1 import Hex1
    from RUC_Generator.Hexagonal.Hex2 import Hex2
    from RUC_Generator.Hexagonal.Hex3 import Hex3
    from RUC_Generator.Square.Square1 import Square1
    from RUC_Generator.Square.Square2 import Square2
    from RUC_Generator.Square.Square3 import Square3
    from RUC_Generator.Random.RandomSBD import RandomSBD
    from RUC_Generator.Random.RandomOptimizationSBD import RandomOptimizationSBD

    # Color Options
    if key == 'Fiber' or key == 'Matrix':
        color_list = ["white", "black", "red", "green", "blue", "yellow", "purple"]
        if key == 'Fiber':
            def_color = 'blue'
        else:
            def_color = 'red'

        return color_list, def_color

    # Ordered microstructure list
    if key == "OrderedList":
        ordered_list = [
                        "Hexagonal", 
                        "Square",
                        ]
        return ordered_list
    
    # Hexagonal pack options
    if key == 'Hexagonal':
        # Create default values
        def_vals_ord = {
                    # Col   Type        Step    Min     Max                         Default Display Name    
                'VF':[1,    'float',    0.001,  0.,     math.pi / (2*math.sqrt(3)), 0.6,    'VF'    ],
                'R' :[2,    'float',    0.001,  0.,     None,                       10.,    'R'     ],
                'NB':[1,    'int',      1,      1,      None,                       10,     'NB'    ],
                'NG':[2,    'int',      1,      1,      None,                       10,     'NG'    ],
                'F' :[1,    'int',      1,      1,      None,                       1,      'F'     ],
                'M' :[2,    'int',      1,      1,      None,                       2,      'M'     ],
                }

        # Create defintion list
        def_list_ord = {
                    "Volume Fraction & Subcell Dimensions":{
                                                            'Inputs':['VF','NB','F','M'],
                                                            'Function':Hex1
                                                            }, 
                    "Volume Fraction & Radius":{
                                                'Inputs':['VF','R','F','M'],
                                                'Function':Hex2
                                                }, 
                    "Radius & Subcell Dimensions":{
                                                    'Inputs':['R','NB','F','M'],
                                                    'Function':Hex3
                                                    }, 
                    }
        
        return def_vals_ord, def_list_ord
    
    # Square pack options
    if key == 'Square':
        # Create default values
        def_vals_ord = {
                    # Col   Type        Step    Min     Max             Default Display Name   
                'VF':[1,    'float',    0.001,  0.,     math.pi / 4,    0.6,    'VF'    ],
                'R' :[2,    'float',    0.001,  0.,     None,           10.,    'R'     ],
                'NB':[1,    'int',      1,      1,      None,           10,     'NB'    ],
                'NG':[2,    'int',      1,      1,      None,           10,     'NG'    ],
                'F' :[1,    'int',      1,      1,      None,           1,      'F'     ],
                'M' :[2,    'int',      1,      1,      None,           2,      'M'     ],
                }

        # Create defintion list
        def_list_ord = {
                    "Volume Fraction & Subcell Dimensions":{
                                                            'Inputs':['VF','NB','F','M'],
                                                            'Function':Square1
                                                            }, 
                    "Volume Fraction & Radius":{
                                                'Inputs':['VF','R','F','M'],
                                                'Function':Square2
                                                }, 
                    "Radius & Subcell Dimensions":{
                                                    'Inputs':['R','NB','F','M'],
                                                    'Function':Square3
                                                    }, 
                    }
        
        return def_vals_ord, def_list_ord
    
    # Random Algorithm Functions
    if key == "AlgorithmList":
        algo_list = [
                        "Soft Body Dynamics", 
                        ]
        return algo_list

    # Soft Body Dynamics Options
    if key == 'SBD':

        # Create default values
        def_vals_rand = {
                            # Col   Type        Step    Min     Max     Default Display Name                Format
                'VF'        :[1,    'float',    0.001,  0.,     0.99,   0.6,    'VF',                       "%.3f"  ],
                'N_fibers'  :[2,    'int',      1,      1,      None,   16,     'Number of Fibers',         "%d"    ],
                'W'         :[1,    'int',      1,      1,      None,   100,    'NB',                       "%d"    ],
                'H'         :[2,    'int',      1,      1,      None,   100,    'NG',                       "%d"    ],
                'k'         :[1,    'float',    0.1,    0.0,    None,   5000.,  'Stiffness (k)',            "%.3f"  ],
                'dt'        :[2,    'float',    1.0e-6, 1.0e-6, None,   0.01,   '\u0394t',                  "%.6f"  ],
                'damping'   :[1,    'float',    1.0e-3, 0.,     0.999,  0.5,    'Damping (c)',              "%.3f"  ],
                'gamma'     :[2,    'float',    1.0e-3, 0.,     None,   1.0,    'Friction Coefficient',     "%.3f"  ],
                'steps'     :[1,    'int',      1,      1,      None,   10000,  'Maxmimum Iteratiaons',     "%d"    ],
                'min_gap'   :[2,    'int',      1,      0,      None,   1,      'Minimum Gap Between Fiber',"%d"    ],
                'n_gen'     :[1,    'int',      1,      1,      None,   1,      'Number of microstructures',"%d"    ],
                            # Col   Type    List            Display Name 
                'periodic'  :[2,    'disc', [True, False], 'Periodic'],
                }
        
        # Define Algorithm Function
        func_rand = RandomSBD

        return def_vals_rand, func_rand

    # Soft Body Dynamics Optimization
    if key == 'SBD_Opt':    
        # Create default input values
        input_space = {
                            # Type      Step    Min     Max     Default Min Default Max Display Name                Format
                'k'         :['float',  0.1,    1.e0,   1.e6,   1.e1,       1.e4,       'Stiffness (k)',            "%.3f"  ],
                'damping'   :['float',  1.0e-3, 0.001,  0.999,  0.2,        0.8,        'Damping (c)',              "%.3f"  ],
                'gamma'     :['float',  1.0e-3, 0.,     1.e3,   1.e-2,      1.e2,       'Friction Coefficient',     "%.3f"  ],
                'seed'      :['float',  1.,     1.,     1.e8,    1.,        1.e6,       'Seed',                     "%.1f"  ]
                
                }
        
        # Create default constant values
        constants = {
                            # Col   Type        Step    Min     Max     Default Display Name                Format
                'W'         :[1,    'int',      1,      1,      None,   100,    'NB',                       "%d"    ],
                'H'         :[2,    'int',      1,      1,      None,   100,    'NG',                       "%d"    ],
                'N_fibers'  :[1,    'int',      1,      1,      None,   16,     'Number of Fibers',         "%d"    ],
                'min_gap'   :[2,    'int',      1,      0,      None,   1,      'Minimum Gap Between Fiber',"%d"    ],
                }
        
        # Create default optimization setting values
        opt_settings = {
                                # Col   Type        Step    Min     Max     Default Display Name         Format
                'batch size'    :[1,    'int',      1,      1,      None,   10,    'Batch Size',        "%d"    ],
                'patience'      :[2,    'int',      1,      1,      None,   25,    'Patience',          "%d"    ],
                'max_iter'      :[1,    'int',      1,      1,      None,   10000, 'Maximum Steps',     "%d"    ],
                'nbins'         :[2,    'int',      1,      2,      None,   10,    'Number of Bins',    "%d"    ],
                }
        
        return input_space, constants, opt_settings
    
    # Soft Body Dynamics Optimization Run
    if key == 'SBD_Opt_Run':

        # Get RVE Settings
        rve_data = {
                    'W':vars[f"opt_rve_W_{tag}"],
                    'H':vars[f"opt_rve_H_{tag}"],
                    'Nfibers':vars[f"opt_rve_N_fibers_{tag}"],
                    'min_gap':vars[f"opt_rve_min_gap_{tag}"]
                    } 

        # Get Optimization Settings 
        opt_settings = {
                    'batch size':vars[f"opt_opt_batch size_{tag}"],
                    'patience':vars[f"opt_opt_patience_{tag}"],
                    'steps':vars[f"opt_opt_max_iter_{tag}"],
                    'nbins':vars[f"opt_opt_nbins_{tag}"],
                    } 
        
        # Get Input Space 
        input_space = {}
        for i, key in enumerate(vars[f"input_space_{tag}"].keys()):
            if vars[f"optin_check_{i}_{tag}"]:
                input_space[key] = [vars[f"optin_num_{i}_low_{tag}"],vars[f"optin_num_{i}_high_{tag}"]] 

        # Get the mask
        try:
            mask, __ = vars[f'mask_{tag}']
        except:
            mask = vars[f'mask_{tag}']

        # Create input dictionary
        func_vars = {
                    'mask':mask,
                    'rve_data':rve_data,
                    'input_space':input_space,
                    'opt_settings':opt_settings,
                    }
        
        # Get the function
        func = RandomOptimizationSBD

        return func, func_vars