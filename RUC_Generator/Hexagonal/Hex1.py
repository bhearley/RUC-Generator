def Hex1(VF, NB, F, M, VI=None, RI=None, I=None):
    """
    Generate a hexagonal pack microstructure with optional interface region.
    The interface can be specified either by thickness (RI) or volume fraction (VI).

    Arguments:
        VF  float   desired fiber volume fraction
        NB  int     number of subcells in the beta direction
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

    # --- Ensure NB is even ---
    nx = NB
    if nx % 2 != 0:
        nx -= 1

    # Hexagonal aspect ratio
    ny = 2 * round((np.sqrt(3) * nx) / 2)

    # --- Fiber radius ---
    R = np.sqrt(((ny*ny/np.sqrt(3)) * VF) / (2 * np.pi))

    # --- Compute interface outer radius ---
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

    # --- Create base RUC (half) ---
    base = M * np.ones((nx//2, ny//2), dtype=int)

    for i in range(nx//2):
        for j in range(ny//2):
            x = i + 0.5
            y = j + 0.5
            # Distance from bottom-left and top-right corners
            dist1 = np.sqrt(x**2 + y**2)
            dist2 = np.sqrt((nx/2 - x)**2 + (ny/2 - y)**2)

            # Fiber assignment
            if dist1 <= R or dist2 <= R:
                base[i, j] = F
            # Interface assignment
            elif (dist1 <= R_outer or dist2 <= R_outer):
                if VI is not None or RI is not None:
                    base[i, j] = I

    # Mirror to create full RUC
    base2 = np.flipud(base)
    base12 = np.vstack([base, base2])
    base13 = np.fliplr(base12)
    mask = np.hstack([base12, base13])
    mask = mask.T

    # --- Output dictionary ---
    out = {
        'VF': np.sum(mask == F) / (nx * ny),
        'R': np.sum(mask[:, int(nx/2)] == F) / 2,
        'NB': len(mask[0]),
        'NG': len(mask),
        'F': F,
        'M': M,
        'VI': np.sum(mask == I) / (nx * ny) if (VI is not None or RI is not None) else 0,
        'RI': np.sum(mask[:, int(nx/2)] == I) / 2 if (VI is not None or RI is not None) else 0,
        'I': I
    }

    return mask, out
