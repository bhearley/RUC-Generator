def RandomCharacterization(mask, nbins = 10):
    import cv2
    import numpy as np
    from scipy.spatial import Delaunay
    from skimage.filters import threshold_multiotsu
    from collections import defaultdict

    # ------------------------------
    # Separate fibers and get centers
    # ------------------------------
    def separate_fibers(mask, min_circular_coverage = 0):
        """
        Separates fibers in a mask and computes centers.
        - Watershed split for irregular fibers.
        - Boundary fibers included if mostly circular.
        - Ignores fibers that are extreme outliers in area.

        Returns:
            final_label_map: labeled fiber mask
            centers: array of fiber centers (x, y)
        """
        def circle_coverage(xs, ys):
            """Fit circle and compute angular coverage in degrees."""
            if len(xs) < 3:
                return 0, None, None
            A = np.c_[2*xs, 2*ys, np.ones_like(xs)]
            b = xs**2 + ys**2
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc, c = sol
            r = np.sqrt(c + xc**2 + yc**2)
            angles = np.arctan2(ys - yc, xs - xc)
            angles = np.sort(angles)
            gaps = np.diff(np.concatenate([angles, [angles[0] + 2*np.pi]]))
            max_gap = gaps.max()
            coverage = 360 * (1 - max_gap / (2*np.pi))
            return coverage, (xc, yc), r

        H, W = mask.shape
        fiber = (mask == 1).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(fiber, connectivity=4)



        centers = []
        for fid in range(1, num_labels):
            ys, xs = np.where(labels == fid)
            if len(xs) == 0:
                continue

            # centroid calculation
            centers.append((xs.mean(), ys.mean()))

        centers = np.array(centers)
        return labels, centers

    # ------------------------------
    # Delaunay triangulation & local VF
    # ------------------------------
    def delaunay_and_vf(mask, centers):
        H, W = mask.shape
        ys, xs = np.indices(mask.shape)
        subcell_centers = np.stack([xs.ravel(), ys.ravel()], axis=1)

        tri = Delaunay(centers)
        local_vf = []

        for simplex in tri.simplices:
            tri_pts = centers[simplex].astype(np.float32)  # shape (3,2)
            
            # Efficient point-in-triangle check using cv2
            pts = subcell_centers.astype(np.float32)
            mask_inside = np.array([cv2.pointPolygonTest(tri_pts, (float(pt[0]), float(pt[1])), False) >= 0
                                    for pt in pts])
            if mask_inside.sum() == 0:
                vf = 0.0
            else:
                pts_idx = np.where(mask_inside)[0]
                fiber_subcells = (mask[subcell_centers[pts_idx,1].astype(int),
                                        subcell_centers[pts_idx,0].astype(int)] == 1).sum()
                vf = fiber_subcells / len(pts_idx)
            local_vf.append(vf)

        return tri, local_vf
    
    # ------------------------------
    # Local VF Statisitcs
    # ------------------------------
    def identify_fc_mrc(mask, tri, centers, local_vf, nbins = 20, n_smooth_iter=0):

        # Enforce array for local volume fraction
        local_vf = np.array(local_vf)

        # Compute statistics
        mean_vf = np.mean(local_vf)
        q1 = np.percentile(local_vf, 25)
        q3 = np.percentile(local_vf, 75)
        iqr = q3 - q1

        counts, bin_edges = np.histogram(local_vf, bins=nbins, range=(0,1), density=False)
    
        # Normalize to get PDF
        pdf = counts / counts.sum()
        
        # Bin centers for plotting
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

        # Threshold into 3 bins using Otsu method
        bins = threshold_multiotsu(local_vf, classes=3)  # returns 2 thresholds

        # Assign each triangle to a bin: 0=sparse, 1=middle, 2=dense
        bin_indices = np.digitize(local_vf, bins)

        # Build neighbors dictionary
        tri_neighbors = defaultdict(list)
        for i, simplex_i in enumerate(tri.simplices):
            for j, simplex_j in enumerate(tri.simplices):
                if i >= j:
                    continue
                if len(set(simplex_i) & set(simplex_j)) == 2:
                    tri_neighbors[i].append(j)
                    tri_neighbors[j].append(i)

        # Perform smoothing
        for _ in range(n_smooth_iter):
            new_bins = bin_indices.copy()
            for i, b in enumerate(bin_indices):
                counts = np.bincount([bin_indices[n] for n in tri_neighbors[i]], minlength=3)
                # Remove rule: too many neighbors not in same bin
                if counts[b] < 2:
                    new_bins[i] = 1  # assign to middle as neutral
                # Add rule: neighbors in other bin
                for bin_val in range(3):
                    if bin_val != b and counts[bin_val] >= 2:
                        new_bins[i] = bin_val
            bin_indices = new_bins

        # Assign fiber clusters and matrix rich clusters
        fc_triangles = np.where(bin_indices == 2)[0]  # dense
        mrc_triangles = np.where(bin_indices == 0)[0] # sparse

        # Get cdf
        cdf = np.cumsum(pdf)

        # Compute the median
        idx = np.searchsorted(cdf, 0.5)

        if idx == 0:
            median_vf = bin_centers[0]
        else:
            # Linear interpolation within the bin
            cdf_low = cdf[idx-1]
            cdf_high = cdf[idx]
            fraction = (0.5 - cdf_low) / (cdf_high - cdf_low)
            median_vf = bin_centers[idx-1] + fraction * (bin_centers[idx] - bin_centers[idx-1])

        return mean_vf, median_vf, iqr, bin_centers, pdf, bin_indices, fc_triangles, mrc_triangles

    
    # ------------------------------
    # Execute
    # ------------------------------
    label_map, centers = separate_fibers(mask)
    tri, local_vf = delaunay_and_vf(mask, centers)
    mean_vf, median_vf, iqr_vf, bin_centers, pdf, bin_indices, fc_triangles, mrc_triangles = identify_fc_mrc(mask, tri, centers, local_vf, nbins = nbins, n_smooth_iter=1)

    return mean_vf, median_vf, iqr_vf, bin_centers, pdf, centers, tri, local_vf

