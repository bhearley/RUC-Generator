def Hex2(VF, R, F, M):
    """
    Generate a hexagonal pack microstructure by defining the volume fraction and radius in subcells.

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
    n = np.sqrt((2*np.pi*R**2)/(VF*np.sqrt(3)))
    nx = n*1/2
    ny = n*np.sqrt(3)/2

    # Define Circle Centers
    centers = [
                [0,0],
                [n*1/2, n*np.sqrt(3)/2],
                [n*-1/2,n* np.sqrt(3)/2],
                [n*1/2, n*-np.sqrt(3)/2],
                [n*-1/2,n* -np.sqrt(3)/2],
            ]
    for i, center in enumerate(centers):
        centers[i] = np.array(center) + np.array([nx, ny])

    # Define Bounding Box
    xmin = 0
    xmax = 2*nx
    ymin = 0
    ymax = 2*ny

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