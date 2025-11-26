def Square1(VF, NB, F, M):
    """
    Generate a square pack microstructure by defining the volume fraction and subcell dimensions.

    Arguments:
        VF  float   desired volume fraction
        NB  int     number of subcells in the beta direction
        F   int     material ID of the fiber
        M   int     material ID of the matrix

    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties
    """
    # Import Modules
    import numpy as np

    # Force even number of subcells
    nx = NB
    if nx % 2 != 0:
        nx -= 1
    ny = NB
    center = [nx/2, ny/2]

    # Fiber radius
    R = np.sqrt(nx**2 * VF / (np.pi))
    
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
    out['NB'] = len(mask[0])
    out['NG'] = len(mask)

    return mask, out