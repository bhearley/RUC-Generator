def Hex2(VF, R, F, M, VI=None, RI=None, I=None):
    """
    Generate a hexagonal pack microstructure with optional interface region.
    The interface can be specified either by thickness (RI) or volume fraction (VI).

    Arguments:
        VF  float   desired fiber volume fraction
        R   float   radius of the fiber in subcells
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

    # --- Validate input ---
    if VI is not None and RI is not None:
        raise ValueError("Specify either VI or RI, not both.")
    if I is None:
        I = 3

    # --- Calculate the spacing vector ---
    n = np.sqrt((2 * np.pi * R**2) / (VF * np.sqrt(3)))
    nx = n * 1/2
    ny = n * np.sqrt(3)/2

    # Define circle centers
    centers = [
        [0,0],
        [n*1/2, n*np.sqrt(3)/2],
        [n*-1/2, n*np.sqrt(3)/2],
        [n*1/2, n*-np.sqrt(3)/2],
        [n*-1/2, n*-np.sqrt(3)/2],
    ]
    for i, c in enumerate(centers):
        centers[i] = np.array(c) + np.array([nx, ny])

    # --- Bounding box and grid ---
    xmin, xmax = 0, 2*nx
    ymin, ymax = 0, 2*ny
    nx = int(round(xmax - xmin))
    ny = int(round(ymax - ymin))
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymin + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)

    # --- Start with matrix ---
    mask = M * np.ones((ny, nx), dtype=int)

    # --- Interface outer radius ---
    if VI is not None:
        A_total = nx * ny
        A_interface = VI * A_total / len(centers)  # distribute interface over fibers
        R_outer = np.sqrt(R**2 + A_interface / np.pi)
        RI_thickness = R_outer - R
    elif RI is not None:
        R_outer = R + RI
        RI_thickness = RI
    else:
        R_outer = R
        RI_thickness = 0

    # --- Fill fibers and interface ---
    for c in centers:
        r2 = (X - c[0])**2 + (Y - c[1])**2
        # Fiber
        mask[r2 <= R**2] = F
        # Interface
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
