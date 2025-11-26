def RandomSBD(W, H, N_fibers, VF, damping, k, dt, steps = 50000, gamma = 1.0, mass = 1.0, min_gap = 1, n_gen = 1, periodic = True, seed = None):
    """
    Generate a random microstructure using soft body dynamics

    Arguments:
        W           int         desired width of the RUC
        H           int         desired height of the RUC
        N_fibers    int         number of fibers in the RUC
        VF          float       desired volume fraction
        damping     float       damping coefficient (between 0 and 1)
        k           float       particle stiffness
        dt          float       time step
        steps       int         number of steps
        gamma       float       friction coefficient
        min_gap     int         minimum allowed gap between fibers
        n_gen       int         number of microstructures to generate
        periodic    boolean     flag to enforce periodicity
    Outputs:
        masks       list        for each generated microstructure, contains the
                                2D array mask and output dictionary
    """

    # Import Modules
    import numpy as np

    if seed is not None:
        np.random.seed(int(seed)) 

    # Function to initialize fiber positions
    def initialize_fibers(W, H, N, VF):
        """Randomly place N fibers with radius computed from VF."""
        radius = np.sqrt((VF * W * H) / (np.pi * N))
        centers = np.random.rand(N, 2) * np.array([W, H])
        return centers, radius

    # function to run soft body dynamics simulation
    def soft_particle_md_periodic(
                                centers, 
                                radius, 
                                W, 
                                H, 
                                damping=0.9,
                                gamma = 1.0, 
                                dt=0.01,
                                steps=50_000, 
                                k=1000.0, 
                                mass=1.0, 
                                min_gap = 1,
                                periodic = True,
                                v_tol = 1e-6
                                ):
        """
        Soft-force, force-based MD simulation for fibers in a periodic box.

        centers : ndarray(N,2) initial positions
        radius : float
        W, H : box dimensions
        damping : velocity damping per step
        dt : time step
        steps : number of time steps
        k : stiffness of repulsive force
        mass : particle mass
        """
        N = centers.shape[0]
        velocities = (np.random.rand(N,2) - 0.5) * 2   # random initial velocity

        for step in range(steps):
            forces = np.zeros_like(centers)

            # ---------- Fiber–fiber forces ----------
            for i in range(N):
                for j in range(i+1, N):

                    dx = centers[j,0] - centers[i,0]
                    dy = centers[j,1] - centers[i,1]

                    # periodic minimum-image convention
                    if periodic:
                        if dx > W/2: dx -= W
                        if dx < -W/2: dx += W
                        if dy > H/2: dy -= H
                        if dy < -H/2: dy += H

                    dist = np.hypot(dx, dy)
                    min_dist = 2*radius + min_gap
                    overlap = min_dist - dist

                    if overlap > 0:
                        # normalized collision direction
                        if dist > 0:
                            nx, ny = dx/dist, dy/dist
                        else:
                            nx, ny = np.random.rand(2)-0.5
                            nrm = np.hypot(nx, ny)
                            nx, ny = nx/nrm, ny/nrm

                        # --- Repulsive force ---
                        f = k * overlap
                        fx, fy = f * nx, f * ny
                        forces[i,0] -= fx
                        forces[i,1] -= fy
                        forces[j,0] += fx
                        forces[j,1] += fy

                        # --- Collision damping (stickiness) ---
                        if damping > 0:
                            # relative velocity
                            dvx = velocities[j,0] - velocities[i,0]
                            dvy = velocities[j,1] - velocities[i,1]

                            # normal component of relative velocity
                            vrel = dvx * nx + dvy * ny

                            if vrel > 0:  # only damp if they are separating/compressing along normal
                                reduction = damping * vrel

                                # apply opposite impulses
                                velocities[i,0] += reduction * nx
                                velocities[i,1] += reduction * ny
                                velocities[j,0] -= reduction * nx
                                velocities[j,1] -= reduction * ny

            # ---------- Drag / friction ----------
            # F_drag = -gamma * v
            forces -= gamma * velocities

            # ---------- Update velocities ----------
            velocities += (forces / mass) * dt

            # ---------- Check convergence ----------
            if np.all(np.linalg.norm(velocities, axis=1) < v_tol):
                # velocities are essentially zero
                break

            # ---------- Update positions ----------
            centers += velocities * dt

            # ---------- Boundary conditions ----------
            if periodic:
                centers[:,0] %= W
                centers[:,1] %= H
            else:
                # clip to walls (soft walls optional)
                centers[:,0] = np.clip(centers[:,0], radius+min_gap, W-radius-min_gap)
                centers[:,1] = np.clip(centers[:,1], radius+min_gap, H-radius-min_gap)

        total_overlap_area = 0.0
        R = radius

        for i in range(N):
            for j in range(i+1, N):

                dx = centers[j,0] - centers[i,0]
                dy = centers[j,1] - centers[i,1]

                # periodic minimum image
                if periodic:
                    if dx > W/2: dx -= W
                    if dx < -W/2: dx += W
                    if dy > H/2: dy -= H
                    if dy < -H/2: dy += H

                d = np.hypot(dx, dy)

                # physical overlap uses only 2R, not min_gap
                if d < 2*R:
                    # circle-circle overlap area
                    A = (
                        2 * R*R * np.arccos(d / (2*R))
                        - 0.5 * d * np.sqrt(max(4*R*R - d*d, 0.0))
                    )
                    total_overlap_area += A
        overlap_pct = total_overlap_area/(W*H)

        return centers, overlap_pct


    # Function to voxelate the microstructure
    def voxelate_periodic_rve(centers, radius, W, H):
        """
        Create a voxelated mask for a periodic RVE.
        1 = fiber, 2 = matrix
        """
        y, x = np.indices((H, W))
        mask = np.full((H, W), 2, dtype=np.uint8)  # start as matrix

        for cx, cy in centers:
            # central fiber
            mask[((x - cx)**2 + (y - cy)**2) <= radius**2] = 1

            # periodic images
            if periodic:
                for dx in [-W, 0, W]:
                    for dy in [-H, 0, H]:
                        if dx == 0 and dy == 0:
                            continue
                        mask[((x - (cx+dx))**2 + (y - (cy+dy))**2) <= radius**2] = 1

        return mask
    
    # Generate multiple RUCs
    masks = []
    for i in range(n_gen):        
        # Initialize position
        centers, radius = initialize_fibers(W, H, N_fibers, VF)

        # Run simulation
        centers_final, overlap_pct = soft_particle_md_periodic(
                                                            centers, 
                                                            radius, 
                                                            W, 
                                                            H, 
                                                            damping=damping,
                                                            gamma = gamma, 
                                                            dt=dt,
                                                            steps=steps, 
                                                            k=k, 
                                                            mass=mass, 
                                                            min_gap = min_gap,
                                                            periodic = periodic,
                                                            )

        # Voxelate
        mask = voxelate_periodic_rve(centers_final, radius, W, H)

        # Calculate actual values
        out = {
                'VF':None,
            'R':None,
            'NB':None,
            'NG':None,
            'F':1,
            'M':2,
            'Overlap':None
            }
        
        # Calculate Volume Fraction
        out['VF'] = np.sum(mask == 1) / (W * H)

        # Calculate Radius
        out['R'] = radius

        # Calculate subcell dimensions
        out['NB'] = W
        out['NG'] = H

        # Define Overlap
        out['Overlap'] = overlap_pct

        # Add to list
        masks.append((f'RVE {i+1}',mask, out))

    return masks

    
