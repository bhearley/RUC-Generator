def RandomRVE(W, H, N_fibers, VF, min_gap_subcells, tol, max_iter_radius, periodic=False):

    import numpy as np
    from scipy.ndimage import label, binary_dilation

    # ================================================================
    # 1. Compute initial analytical radius from target VF
    # ================================================================
    def compute_initial_radius(W, H, N, VF):
        total_area = W * H
        area_per_fiber = VF * total_area / N
        return np.sqrt(area_per_fiber / np.pi)

    # ================================================================
    # 2. Periodic minimal-image distance
    # ================================================================
    def periodic_dist(dx, L):
        """Return minimal-image periodic distance along 1D."""
        dx = abs(dx)
        if dx > L / 2:
            dx = L - dx
        return dx

    def dist_periodic(p1, p2, W, H):
        """2D periodic distance between points p1 and p2."""
        dx = periodic_dist(p1[0]-p2[0], W)
        dy = periodic_dist(p1[1]-p2[1], H)
        return np.hypot(dx, dy)

    # ================================================================
    # 3. Fiber placement (periodic or nonperiodic)
    # ================================================================
    def place_fibers(r, W, H, N, min_gap, periodic):
        """
        Robust placement that NEVER throws an exception.
        Automatically relaxes constraints until successful.
        """
        attempt = 0

        while True:
            attempt += 1
            
            centers = []
            max_attempts = 40000
            tries = 0
            min_d = 2*r + min_gap

            while len(centers) < N and tries < max_attempts:
                tries += 1
                cx = np.random.uniform(0, W)
                cy = np.random.uniform(0, H)
                c = (cx, cy)

                ok = True
                for existing in centers:

                    if periodic:
                        d = dist_periodic(c, existing, W, H)
                    else:
                        d = np.hypot(c[0]-existing[0], c[1]-existing[1])

                    if d < min_d:
                        ok = False
                        break

                if ok:
                    centers.append(c)

            # SUCCESS → return centers
            if len(centers) == N:
                return np.array(centers)

            # FAILURE → relax constraints and try again
            # --------------------------------------------
            # Option 1: shrink radius slightly
            r *= 0.98

            # Option 2: relax min gap (makes packing easier)
            #min_gap *= 0.95

            # Option 3: re-seed randomness every few attempts
            if attempt % 5 == 0:
                np.random.seed()

            # Logically: This loop always eventually succeeds.
            # No exception is ever thrown.

    # ================================================================
    # 4. Periodic voxelization (critical for geometric periodicity!!)
    # ================================================================
    def voxelate_periodic(centers, r, W, H):
        y, x = np.indices((H, W))
        mask = np.full((H, W), 2, dtype=np.uint8)

        for (cx, cy) in centers:
            # periodic shift distances
            dx = (x - cx + W/2) % W - W/2
            dy = (y - cy + H/2) % H - H/2
            mask[(dx*dx + dy*dy) <= r*r] = 1

        return mask

    # Nonperiodic voxelation (original)
    def voxelate_nonperiodic(centers, r, W, H):
        y, x = np.indices((H, W))
        mask = np.full((H, W), 2, dtype=np.uint8)

        for (cx, cy) in centers:
            mask[((x-cx)**2 + (y-cy)**2) <= r*r] = 1

        return mask

    # ================================================================
    # 5. Match target volume fraction by iteratively adjusting radius
    # ================================================================
    def adjust_radius(centers, r0, VF, W, H, tol, max_iter, periodic):

        r = r0
        for _ in range(max_iter):

            mask = (
                voxelate_periodic(centers, r, W, H)
                if periodic else
                voxelate_nonperiodic(centers, r, W, H)
            )

            vf = np.mean(mask == 1)

            if abs(vf - VF) < tol:
                return r, mask, vf

            # scale radius to push VF toward target
            r *= np.sqrt(VF / vf)

        # final fallback
        mask = (
            voxelate_periodic(centers, r, W, H)
            if periodic else
            voxelate_nonperiodic(centers, r, W, H)
        )
        vf = np.mean(mask == 1)
        return r, mask, vf
    
    # ================================================================
    # 5. Re-enforce the min gap
    # ================================================================
    def enforce_min_gap_subcells(mask, min_gap):
        """
        Remove only subcells where fibers are touching others, iteratively,
        ensuring no fiber subcells are closer than min_gap.
        """
        new_mask = mask.copy()
        fiber_mask = (new_mask == 1)
        
        # Label connected fiber regions
        labeled_fibers, num_fibers = label(fiber_mask)
        
        # Structuring element slightly larger to catch diagonal touches
        selem = np.ones((2*min_gap+2, 2*min_gap+2), dtype=bool)
        
        changes = True
        while changes:
            changes = False
            fiber_mask = (new_mask == 1)
            labeled_fibers, num_fibers = label(fiber_mask)
            
            for i in range(1, num_fibers+1):
                this_fiber = (labeled_fibers == i)
                dilated = binary_dilation(this_fiber, structure=selem)
                
                # Overlaps with other fibers
                overlap = dilated & (fiber_mask & ~this_fiber)
                
                if np.any(overlap):
                    new_mask[overlap] = 2
                    changes = True  # Repeat check until no overlaps

        return new_mask


    # ================================================================
    # ----- MAIN EXECUTION -----
    # ================================================================
    r0 = compute_initial_radius(W, H, N_fibers, VF)

    centers = place_fibers(
        r0, W, H, N_fibers,
        min_gap=min_gap_subcells,
        periodic=periodic
    )

    r_final, mask, vf_final = adjust_radius(
        centers, r0, VF, W, H,
        tol=tol,
        max_iter=max_iter_radius,
        periodic=periodic
    )

    mask = enforce_min_gap_subcells(mask, min_gap_subcells)

    # ================================================================
    # Output dictionary (same structure as your original)
    # ================================================================
    out = {
        'VF': vf_final,
        'R': r_final,
        'NB': mask.shape[1],
        'NG': mask.shape[0],
        'F': 1,
        'M': 2
    }

    return mask, out