def Square3(NB, R, F, M, VI=None, RI=None, I=None):
    """
    Generate a square pack microstructure with optional interface region.
    The interface can be specified either by thickness (RI) or volume fraction (VI).

    Arguments:
        NB  int     number of subcells in each direction
        R   float   radius of the fiber in subcells
        F   int     material ID of the fiber
        M   int     material ID of the matrix
        VI  float   optional interface volume fraction (cannot be used with RI)
        RI  float   optional interface thickness (cannot be used with VI)
        I   int     optional interface material ID (default 3)

    Outputs:
        mask    2D integer array defining the microstructure
        out     dict with microstructure properties
    """

    import numpy as np

    # --- Validate input ---
    if VI is not None and RI is not None:
        raise ValueError("Specify either VI or RI, not both.")
    if I is None:
        I = 3

    # --- Enforce NB large enough for fiber ---
    if NB <= 2 * R:
        NB = int(1.05 * 2 * R)

    # Force even grid
    nx = NB
    if nx % 2 != 0:
        nx -= 1
    ny = nx
    center = [nx/2, ny/2]

    # --- Determine interface outer radius ---
    if VI is not None:
        # Compute thickness from volume fraction
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

    # --- Grid extents ---
    xmin, xmax = 0, nx
    ymin, ymax = 0, ny
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    xs = xmin + (np.arange(nx) + 0.5) * dx
    ys = ymin + (np.arange(ny) + 0.5) * dy
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
