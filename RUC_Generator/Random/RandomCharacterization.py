def RandomCharacterization(mask, nbins = 10):
    import cv2
    import numpy as np
    from scipy.spatial import Delaunay
    from skimage.filters import threshold_multiotsu
    from collections import defaultdict

    # ------------------------------
    # Separate fibers and get centers
    # ------------------------------
    def separate_fibers(mask):
        fiber = (mask == 1).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(fiber, connectivity=4)

        final_label_map = np.zeros_like(labels)
        current_id = 1

        for lbl in range(1, num_labels):
            blob = (labels == lbl).astype(np.uint8)
            if blob.sum() == 0:
                continue

            cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnts) == 0:
                continue
            cnt = cnts[0]
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
            needs_watershed = circularity < 0.75

            if not needs_watershed:
                final_label_map[labels == lbl] = current_id
                current_id += 1
                continue

            # Watershed split
            dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 1, 0)
            sure_fg = sure_fg.astype(np.uint8)
            sure_bg = cv2.dilate(blob, np.ones((3,3), np.uint8), iterations=3)
            unknown = sure_bg - sure_fg
            n_fg, markers = cv2.connectedComponents(sure_fg)
            markers[unknown == 1] = 0
            blob_color = np.dstack([blob*255]*3)
            markers = cv2.watershed(blob_color, markers)
            for uw in np.unique(markers[markers > 1]):
                final_label_map[markers == uw] = current_id
                current_id += 1

        centers = []
        for fid in range(1, current_id):
            ys, xs = np.where(final_label_map == fid)
            if len(xs) == 0:
                continue
            centers.append((xs.mean(), ys.mean()))

        centers = np.array(centers)
        return final_label_map, centers

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

        return mean_vf, iqr, bin_centers, pdf, bin_indices, fc_triangles, mrc_triangles

    
    # ------------------------------
    # Execute
    # ------------------------------
    label_map, centers = separate_fibers(mask)
    tri, local_vf = delaunay_and_vf(mask, centers)
    mean_vf, iqr_vf, bin_centers, pdf, bin_indices, fc_triangles, mrc_triangles = identify_fc_mrc(mask, tri, centers, local_vf, nbins = nbins, n_smooth_iter=1)

    return mean_vf, iqr_vf, bin_centers, pdf, centers, tri, local_vf

