def Square2(VF, R, F, M):
    """
    Generate a square pack microstructure by defining the volume fraction and radius in subcells.

    Arguments:
        VF  float   desired volume fraction
        R   float   radius of the fiber in subcells
        F   int     material ID of the fiber
        M   int     material ID of the matrix
    
    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties

    """

    # Import modules
    import numpy as np

    # Calculate the spacing vector length
    nx = np.sqrt((np.pi*R**2/VF))
    ny = nx
    center = [nx/2, ny/2]

    # Define Bounding Box
    xmin = 0
    xmax = nx
    ymin = 0
    ymax = ny

    # Define grid resolution
    nx = int(round(xmax - xmin))
    ny = int(round(ymax - ymin))

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny


    # Create subcell centers
    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymin + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)

    # Integer grid: start all zeros
    mask = M * np.ones((ny, nx), dtype=int)

    # Fill fibers
    inside = (X - center[0])**2 + (Y - center[1])**2 <= R**2
    mask[inside] = F

    # Calculate actual values
    out = {
            'VF':None,
           'R':None,
           'NB':None,
           'NG':None,
           'F':F,
           'M':M
           }
    
    # Calculate Volume Fraction
    out['VF'] = np.sum(mask == F) / (nx * ny)

    # Calculate Radius
    out['R'] = np.sum(mask[:,int(nx/2)] == F)/2

    # Calculate subcell dimensions
    out['NB'] = nx
    out['NG'] = ny

    return mask, out