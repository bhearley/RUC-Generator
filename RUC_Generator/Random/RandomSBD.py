def RandomSBD(W, H, N_fibers, VF, damping, k, dt, steps=50000, gamma=1.0,
              mass=1.0, min_gap=1, v_init=2, n_gen=1, periodic=True, seed=None,
              VI=None, RI=None, I=3):
    """
    Generate a random microstructure using soft body dynamics, with optional interface region.
    
    Arguments:
        W, H        int     RUC width and height
        N_fibers    int     number of fibers
        VF          float   desired fiber volume fraction
        damping, k, dt, steps, gamma, mass, min_gap, periodic: soft-body dynamics params
        n_gen       int     number of RUCs to generate
        seed        int     random seed
        VI          float   optional interface volume fraction (per fiber)
        RI          float   optional interface thickness (cannot use with VI)
        I           int     interface material ID (default 3)
    Outputs:
        masks       list    tuples of (name, mask, out_dict) per RUC
    """
    import numpy as np

    if seed is not None:
        np.random.seed(int(seed))

    if VI is not None and RI is not None:
        raise ValueError("Specify either VI or RI, not both.")

    # --- Initialize fibers ---
    def initialize_fibers(W, H, N, VF):
        radius = np.sqrt((VF * W * H) / (np.pi * N))
        centers = np.random.rand(N, 2) * np.array([W, H])
        return centers, radius

    # --- Soft-body dynamics simulation ---
    # (same as your original function, unchanged)
    def soft_particle_md_periodic(centers, radius, W, H, damping=0.9,
                                  gamma=1.0, dt=0.01, steps=50000,
                                  k=1000.0, mass=1.0, v_init =2, min_gap=1, periodic=True,
                                  v_tol=1e-6):
        N = centers.shape[0]
        velocities = (np.random.rand(N, 2) - 0.5) * 2
        for step in range(steps):
            forces = np.zeros_like(centers)
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[j,0] - centers[i,0]
                    dy = centers[j,1] - centers[i,1]
                    if periodic:
                        if dx > W/2: dx -= W
                        if dx < -W/2: dx += W
                        if dy > H/2: dy -= H
                        if dy < -H/2: dy += H
                    dist = np.hypot(dx, dy)
                    if RI is not None:
                        R_eff = radius + RI
                    else:
                        R_eff = radius

                    # minimum center distance for no overlap
                    min_dist = 2 * R_eff + min_gap
                    overlap = min_dist - dist
                    if overlap > 0:
                        if dist > 0:
                            nx, ny = dx/dist, dy/dist
                        else:
                            nx, ny = (np.random.rand(2)-0.5)*(2*v_init)
                            nx, ny = nx/np.hypot(nx, ny), ny/np.hypot(nx, ny)
                        f = k * overlap
                        fx, fy = f * nx, f * ny
                        forces[i,0] -= fx; forces[i,1] -= fy
                        forces[j,0] += fx; forces[j,1] += fy
                        dvx = velocities[j,0] - velocities[i,0]
                        dvy = velocities[j,1] - velocities[i,1]
                        vrel = dvx*nx + dvy*ny
                        if vrel > 0:
                            reduction = damping * vrel
                            velocities[i,0] += reduction*nx
                            velocities[i,1] += reduction*ny
                            velocities[j,0] -= reduction*nx
                            velocities[j,1] -= reduction*ny
            forces -= gamma * velocities
            velocities += (forces / mass) * dt
            if np.all(np.linalg.norm(velocities, axis=1) < v_tol):
                break
            centers += velocities * dt
            if periodic:
                centers[:,0] %= W
                centers[:,1] %= H
            else:
                centers[:,0] = np.clip(centers[:,0], radius+min_gap, W-radius-min_gap)
                centers[:,1] = np.clip(centers[:,1], radius+min_gap, H-radius-min_gap)
        # Compute total overlap percentage
        total_overlap_area = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[j,0] - centers[i,0]
                dy = centers[j,1] - centers[i,1]
                if periodic:
                    if dx > W/2: dx -= W
                    if dx < -W/2: dx += W
                    if dy > H/2: dy -= H
                    if dy < -H/2: dy += H
                d = np.hypot(dx, dy)
                R_phys = R_eff
                if d < 2 * R_phys:
                    A = (
                        2 * R_phys * R_phys * np.arccos(d / (2 * R_phys))
                        - 0.5 * d * np.sqrt(max(4 * R_phys * R_phys - d*d, 0))
                    )
                    total_overlap_area += A
        overlap_pct = total_overlap_area/(W*H)
        return centers, overlap_pct

    # --- Voxelate microstructure with optional interface ---
    def voxelate_periodic_rve(centers, radius, W, H, RI=None, I=3):
        """
        Voxelate RVE with optional interface.
        
        mask values:
            1 = fiber
            I = interface (if RI is not None)
            2 = matrix
        """
        y, x = np.indices((H, W))
        mask = np.full((H, W), 2, dtype=np.uint8)  # matrix

        # Total radius if interface is present
        if RI is not None:
            total_radius = radius + RI

        # Helper: paint one circle + interface
        def paint_circle(cx, cy):
            # squared distances
            dx2 = (x - cx)**2
            dy2 = (y - cy)**2
            r2 = radius**2

            # fiber region
            fiber_mask = (dx2 + dy2) <= r2
            mask[fiber_mask] = 1

            if RI is not None:
                tr2 = total_radius**2
                outer_mask = (dx2 + dy2) <= tr2

                # interface = outer ring minus fiber
                interface_mask = outer_mask & (~fiber_mask)

                # assign scalar I safely
                mask[interface_mask] = I

        # Paint main fibers
        for cx, cy in centers:
            paint_circle(cx, cy)

            # periodic copies
            if periodic:
                for dx in [-W, 0, W]:
                    for dy in [-H, 0, H]:
                        if dx == 0 and dy == 0:
                            continue
                        paint_circle(cx + dx, cy + dy)

        return mask

    # --- Generate RUCs ---
    masks = []
    for i in range(n_gen):
        centers, radius = initialize_fibers(W, H, N_fibers, VF)

        # Compute RI if VI is given
        if VI is not None:
            A_total = W*H
            A_interface_per_fiber = VI * A_total / N_fibers
            RI_val = np.sqrt(radius**2 + A_interface_per_fiber/np.pi) - radius
        else:
            RI_val = RI

        # Run dynamics
        centers_final, overlap_pct = soft_particle_md_periodic(
            centers, radius, W, H, damping=damping, gamma=gamma, dt=dt,
            steps=steps, k=k, mass=mass, v_init=v_init, min_gap=min_gap, periodic=periodic
        )

        # Voxelate
        mask = voxelate_periodic_rve(centers_final, radius, W, H, RI=RI_val, I=I)

        # Compute outputs
        out = {
            'VF': np.sum(mask==1)/(W*H),
            'VI': np.sum(mask==I)/(W*H) if RI_val is not None else 0,
            'R': radius,
            'RI': RI_val,
            'NB': W,
            'NG': H,
            'F':1,
            'M':2,
            'I': I,
            'Overlap': overlap_pct
        }

        masks.append((f'RVE {i+1}', mask, out))

    return masks