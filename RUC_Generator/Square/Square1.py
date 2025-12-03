def Square1(VF, NB, F, M, VI=None, RI=None, I=None):
    """
    Generate a square pack microstructure by defining the volume fraction and subcell dimensions.
    The interface can be specified either by thickness (RI) or volume fraction (VI).

    Arguments:
        VF  float   desired fiber volume fraction
        NB  int     number of subcells in each direction
        F   int     material ID of the fiber
        M   int     material ID of the matrix
        VI  float   optional interface volume fraction (cannot be used with RI)
        RI  float   optional interface thickness (cannot be used with VI)
        I   int     optional interface material ID (default 3)

    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties
    """

    import numpy as np

    # --- Validate inputs ---
    if VI is not None and RI is not None:
        raise ValueError("Specify either VI or RI, not both.")
    if I is None:
        I = 3

    # --- Ensure NB is even ---
    nx = NB
    if nx % 2 != 0:
        nx -= 1
    ny = nx
    center = [nx/2, ny/2]

    # --- Compute fiber radius for desired VF ---
    R = np.sqrt(nx**2 * VF / np.pi)

    # --- Determine interface outer radius ---
    if VI is not None:
        # Compute outer radius to achieve VI
        A_total = nx * ny
        A_interface = VI * A_total
        R_outer = np.sqrt(R**2 + A_interface / np.pi)
        RI_thickness = R_outer - R
    elif RI is not None:
        R_outer = R + RI
        RI_thickness = RI
    else:
        R_outer = R
        RI_thickness = 0

    # --- Prepare grid ---
    xs = np.arange(nx) + 0.5
    ys = np.arange(ny) + 0.5
    X, Y = np.meshgrid(xs, ys)

    # --- Start with matrix ---
    mask = M * np.ones((ny, nx), dtype=int)

    # --- Fiber region ---
    r2 = (X - center[0])**2 + (Y - center[1])**2
    mask[r2 <= R**2] = F

    # --- Interface region ---
    if VI is not None or RI is not None:
        mask[(r2 > R**2) & (r2 <= R_outer**2)] = I

    # --- Output dictionary ---
    out = {
        'VF': np.sum(mask == F) / (nx * ny),
        'R': np.sum(mask[:, int(nx/2)] == F) / 2,
        'NB': nx,
        'NG': ny,
        'F': F,
        'M': M,
        'VI': np.sum(mask == I) / (nx * ny) if (VI is not None or RI is not None) else 0,
        'RI': np.sum(mask[:, int(nx/2)] == I) / 2 if (VI is not None or RI is not None) else 0,
        'I': I
    }

    return mask, out
