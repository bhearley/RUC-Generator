def RandomRVE(W, H, N_fibers, VF, min_gap_subcells, tol, max_iter_radius):
    
    # Import Modules
    import numpy as np
    
    # ================================================================
    # 1. Compute initial continuous radius from target VF
    # ================================================================
    def compute_initial_radius(W, H, N_fibers, VF):
        total_area = W * H
        area_per_fiber = VF * total_area / N_fibers
        return np.sqrt(area_per_fiber / np.pi)

    # ================================================================
    # 2. Random Non-Overlapping Fiber Placement with Minimum Gap
    # ================================================================
    def generate_centers_with_gap(r, W, H, N_fibers, min_gap_subcells=2, max_attempts=10000):
        """
        Place N_fibers randomly with a minimum matrix gap between fibers.
        min_gap_subcells: minimum number of matrix voxels separating fibers.
        """
        centers = []
        attempts = 0
        while len(centers) < N_fibers and attempts < max_attempts:
            attempts += 1
            cx = np.random.uniform(0.0, W)
            cy = np.random.uniform(0.0, H)
            ok = True
            for x0, y0 in centers:
                # Minimum separation = 2*radius + gap
                if np.hypot(cx - x0, cy - y0) < 2*r + min_gap_subcells:
                    ok = False
                    break
            if ok:
                centers.append((cx, cy))
        if len(centers) < N_fibers:
            raise RuntimeError(f"Could only place {len(centers)} fibers after {attempts} attempts")
        return np.array(centers)

    def generate_valid_centers(r, W, H, N_fibers, min_gap_subcells=2, max_attempts=10000):
        """
        Keep retrying random placement until all fibers are successfully placed.
        """
        success = False
        while not success:
            try:
                centers = generate_centers_with_gap(r, W, H, N_fibers, min_gap_subcells, max_attempts)
                success = True
            except RuntimeError:
                # Retry with new random seed / new random positions
                continue
        return centers

    # ================================================================
    # 3. Voxelate Fibers
    # ================================================================
    def voxelate(centers, r, W, H):
        """
        Converts fiber centers and radius into a 2D mask.
        Fiber = 1, Matrix = 2
        """
        y, x = np.indices((H, W))
        mask = np.full((H, W), 2, dtype=np.uint8)  # matrix = 2
        for (cx, cy) in centers:
            mask[((x - cx)**2 + (y - cy)**2) <= r*r] = 1
        voxel_VF = np.mean(mask == 1)
        return mask, voxel_VF

    # ================================================================
    # 4. Adjust radius iteratively to match target VF
    # ================================================================
    def adjust_radius_to_target(centers, r_initial, VF, W, H, tol=1e-3, max_iter=20):
        r = r_initial
        for _ in range(max_iter):
            mask, voxel_VF = voxelate(centers, r, W, H)
            if abs(voxel_VF - VF) < tol:
                return r, mask, voxel_VF
            # Scale radius based on area ratio
            r *= np.sqrt(VF / voxel_VF)
        # Final attempt
        mask, voxel_VF = voxelate(centers, r, W, H)
        return r, mask, voxel_VF

    # Generate random RVE
    r0 = compute_initial_radius(W, H, N_fibers, VF)
    centers = generate_valid_centers(r0, W, H, N_fibers, min_gap_subcells)
    r_final, mask, voxel_VF = adjust_radius_to_target(centers, r0, VF, W, H, tol, max_iter_radius)

    # Calculate actual values
    out = {
            'VF':None,
            'R': None,
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

    # Calculate Volume Fraction
    out['R'] = r_final

    # Calculate subcell dimensions
    out['NB'] = nx
    out['NG'] = ny

    return mask, out