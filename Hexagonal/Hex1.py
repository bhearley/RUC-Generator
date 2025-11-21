def Hex1(VF, NB, F, M):
    """
    Generate a hexagonal pack microstructure by defining the volume fraction and subcell dimensions.

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

    # Round to nearest integer for aspect ratio
    ny = 2 * round((np.sqrt(3) * nx) / 2)  

    # Fiber radius
    R = np.sqrt(((ny*ny/np.sqrt(3))*VF)/(2*np.pi))

    # Base subcell (half RUC)
    base = M * np.ones((nx//2, ny//2), dtype=int)
    
    for i in range(nx//2):
        for j in range(ny//2):
            x = i + 0.5
            y = j + 0.5
            dist = np.sqrt(x**2 + y**2)
            dist2 = np.sqrt((nx/2 - x)**2 + (ny/2 - y)**2)
            if dist <= R or dist2 <= R:
                base[i, j] = F

    # Mirror to create full RUC
    base2 = np.flipud(base)
    base12 = np.vstack([base, base2])
    base13 = np.fliplr(base12)
    mask = np.hstack([base12, base13])
    mask = mask.T

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