def ReadCSV(content):
    """
    Read either a .csv file containing a RUC definition.

    Arguments:
        content str         content of the .csv file
    
    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties
    """

    # Import Modules
    import numpy as np
    
    # Get mask
    mask = np.array([list(map(int, line.split(','))) for line in content.strip().splitlines()])

    # Calculate actual values
    out = {
            'VF':None,
           'NB':None,
           'NG':None,
           'F':1,
           'M':2
           }
    
    # Set Dimensions
    nx = len(mask[0,:])
    ny = len(mask[:,0])

    # Calculate Volume Fraction
    out['VF'] = np.sum(mask == 1) / (nx * ny)

    # Calculate subcell dimensions
    out['NB'] = nx
    out['NG'] = ny

    return mask, out