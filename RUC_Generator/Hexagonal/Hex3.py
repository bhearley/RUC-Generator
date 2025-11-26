def Hex3(NB, R, F, M):
    """
    Generate a hexagonal pack microstructure by defining the subcell width and radius in subcells.

    Arguments:
        NB  int     number of subcells in the beta direction
        R   float   radius of the fiber in subcells
        F   int     material ID of the fiber
        M   int     material ID of the matrix

    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties
    """

    # Import modules
    import numpy as np

    # Enforce minimum NB
    if NB <= 2*R:
        NB = int(1.05*2*R)

    # Enforce NB be even
    nx = NB
    if nx % 2 != 0:
        nx -= 1
    ny = 2 * round((np.sqrt(3) * nx) / 2)  # nearest integer for aspect ratio

    
    # Define Circle Centers
    centers = [
                [0,0],
                [0, ny],
                [nx, 0],
                [nx, ny],
                [nx/2, ny/2],
            ]


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
    for c in centers:
        inside = (X - c[0])**2 + (Y - c[1])**2 <= R**2
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