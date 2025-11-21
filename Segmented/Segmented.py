def Segmented(Input):
    # Import modules
    import cv2
    import numpy as np
    from scipy import ndimage
    from scipy.optimize import least_squares
    from scipy.signal import correlate2d
    from sklearn.linear_model import RANSACRegressor

    # Function for splitting touching circles
    def split_touching_circles(image, color):

        # Mask the area
        color_cv2 = (color[2], color[1], color[0]) #BGR Format
        mask = cv2.inRange(image, color_cv2, color_cv2) 

        # Morphological cleanup
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Get sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Get sure foreground area (center of objects)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # So background is 1 instead of 0
        markers[unknown == 255] = 0

        # Convert original to BGR if needed and apply watershed
        img_for_ws = image.copy()
        if len(img_for_ws.shape) == 2 or img_for_ws.shape[2] == 1:
            img_for_ws = cv2.cvtColor(img_for_ws, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(img_for_ws, markers)

        # Create output image where each circle has a different color
        output = np.zeros_like(image)
        unique_labels = np.unique(markers)
        np.random.seed(42)
        for label in unique_labels:
            if label <= 1:  # skip background and border
                continue
            mask = markers == label
            color = np.random.randint(0, 255, 3)
            output[mask] = color

        return output, markers
    
    # Function to get contours for each body
    def get_body_boundaries(markers, exclude_labels=[-1, 0, 1]):

        # Preallocate the contours dictionary
        contours_dict = {}

        # Loop through labels
        unique_labels = np.unique(markers)
        for label in unique_labels:
            if label in exclude_labels:
                continue

            # Create binary mask for this label
            mask = np.uint8(markers == label)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                contours_dict[label] = contours[0]  # Largest or first contour

        return contours_dict
    
    #   Function to check if contour is touching the edge of the image
    def is_contour_touching_edge(contour, image_shape, margin=1):

        # Get image shape
        h, w = image_shape[:2]
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]

        # Check for boundary touching
        return (
            np.any(xs <= margin) or
            np.any(xs >= w - 1 - margin) or
            np.any(ys <= margin) or
            np.any(ys >= h - 1 - margin)
        )
    
    #   Function to determine boundary intersection type
    def boundary_intersection_type(contour):

        # Get Arc Points
        arc_points = contour.squeeze()  # shape (N, 2)

        # Fit Cricle
        xc, yc, R, status = fit_circle_least_squares(arc_points)

        # Check for fit
        if status:
            return xc, yc, R, status
        else:
            return None, None, None, False
        
    # Function to fit a cirlce to a set of points using least squares regression
    def fit_circle_least_squares(points):

        def calc_R(xc, yc):
            return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)

        def objective(c):
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        # Initial guess: centroid and mean distance
        x0 = np.mean(points[:, 0])
        y0 = np.mean(points[:, 1])

        # Perform regression
        res = least_squares(objective, x0=[x0, y0])
        xc, yc = res.x
        Ri = calc_R(xc, yc)
        R = Ri.mean()

        return xc, yc, R, res.success

    # Function to find best fit circle for a contour (full circles only)
    def polar_fit_circle(contour):

        # Get centroid of contour
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Convert points to polar coordinates
        pts = contour.reshape(-1, 2)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        theta = np.arctan2(dy, dx)
        r = np.sqrt(dx**2 + dy**2)

        # Regression or outlier rejection - sort by angle to improve fit stability
        sort_idx = np.argsort(theta)
        theta_sorted = theta[sort_idx].reshape(-1, 1)
        r_sorted = r[sort_idx]

        # Use RANSAC to robustly fit r = constant
        ransac = RANSACRegressor()
        ransac.fit(theta_sorted, r_sorted)
        r_fit = ransac.predict(theta_sorted)
        radius = np.mean(r_fit)

        return (cx, cy, radius, theta, r_sorted)

    # Function to fit circles from the watershed markers
    def fit_circles_from_markers(markers, image_shape, min_angular_coverage=np.pi * 1.5, min_coverage_ratio=0.7):

        boundaries = get_body_boundaries(markers)
        fitted_circles = {}

        for label, contour in boundaries.items():
            if is_contour_touching_edge(contour, image_shape):
                x, y, r, status = boundary_intersection_type(contour)
                if status == False:
                    continue

            else:
                result = polar_fit_circle(contour)
                if result is None:
                    continue

                x, y, r, theta, r_values = result

                # Check angular coverage
                theta_unwrapped = np.unwrap(theta)
                ang_span = np.max(theta_unwrapped) - np.min(theta_unwrapped)
                if ang_span < min_angular_coverage:
                    continue

                # Check area coverage
                circle_area = np.pi * r**2
                contour_area = cv2.contourArea(contour)
                coverage_ratio = contour_area / circle_area
                if coverage_ratio < min_coverage_ratio:
                    continue

            fitted_circles[label] = (x, y, r)

        # Assign x and y points for each circle
        fitted_pts = {}
        ct = 0
        npts = 1000
        for fit in fitted_circles:
            fitted_pts[ct] = {
                            'xc':fitted_circles[fit][0],
                            'yc':fitted_circles[fit][1],
                            'R':fitted_circles[fit][2],
                            'x':[],
                            'y':[]
                            }
            theta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
            x = fitted_circles[fit][0] + fitted_circles[fit][2] * np.cos(theta)
            y = fitted_circles[fit][1] + fitted_circles[fit][2] * np.sin(theta)
            fitted_pts[ct]['x'] = x
            fitted_pts[ct]['y'] = y
            ct = ct + 1

        return fitted_circles, fitted_pts
    
    # Function to get intersecting boundaries between fit cirlces
    def get_intersecting_boundaries(fitted_pts):

        # Get Keys
        keys = list(fitted_pts.keys())

        # Loop through all circles and find intersection
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i != j:
                    pts = circle_intersections(
                                                fitted_pts[keys[i]]['xc'], 
                                                fitted_pts[keys[i]]['yc'], 
                                                fitted_pts[keys[i]]['R'], 
                                                fitted_pts[keys[j]]['xc'], 
                                                fitted_pts[keys[j]]['yc'], 
                                                fitted_pts[keys[j]]['R'], 
                                                tol=1e-6
                                                )
                    
                    if len(pts) > 0:

                        # Remove Points from First Circle
                        new_x = []
                        new_y = []
                        for k in range(len(fitted_pts[keys[i]]['x'])):
                            dx = fitted_pts[keys[i]]['x'][k] - fitted_pts[keys[j]]['xc']
                            dy = fitted_pts[keys[i]]['y'][k] - fitted_pts[keys[j]]['yc']
                            if dx**2 + dy**2 > fitted_pts[keys[j]]['R']**2:
                                new_x.append(fitted_pts[keys[i]]['x'][k] )
                                new_y.append(fitted_pts[keys[i]]['y'][k] )
                        fitted_pts[keys[i]]['x'] = new_x
                        fitted_pts[keys[i]]['y'] = new_y

                        # Remove Points from First Circle
                        new_x = []
                        new_y = []
                        for k in range(len(fitted_pts[keys[j]]['x'])):
                            dx = fitted_pts[keys[j]]['x'][k] - fitted_pts[keys[i]]['xc']
                            dy = fitted_pts[keys[j]]['y'][k] - fitted_pts[keys[i]]['yc']
                            if dx**2 + dy**2 > fitted_pts[keys[i]]['R']**2:
                                new_x.append(fitted_pts[keys[j]]['x'][k] )
                                new_y.append(fitted_pts[keys[j]]['y'][k] )
                        fitted_pts[keys[j]]['x'] = new_x
                        fitted_pts[keys[j]]['y'] = new_y

        return fitted_pts
    
    # Function to find circle intersecting points
    def circle_intersections(x0, y0, r0, x1, y1, r1, tol=1e-6):

        # Calculate parameters
        dx = x1 - x0
        dy = y1 - y0
        d = np.hypot(dx, dy)

        # No solution
        if d > r0 + r1 + tol or d < abs(r0 - r1) - tol:
            return []  # no intersection

        # Coincident
        if d < tol and abs(r0 - r1) < tol:
            return None  # infinite intersections

        # Find a, h
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h_sq = r0**2 - a**2
        if h_sq < 0:
            h_sq = 0  # fix numerical error

        h = np.sqrt(h_sq)

        # Point P2 (base point)
        x2 = x0 + a * dx / d
        y2 = y0 + a * dy / d

        # Tangent (one point)
        if np.isclose(h, 0):
            return [(x2, y2)]

        # Two intersection points
        rx = -dy * (h / d)
        ry =  dx * (h / d)

        p1 = (x2 + rx, y2 + ry)
        p2 = (x2 - rx, y2 - ry)

        return [p1, p2]
    
    # Function to generate a new image
    def generate_new_image(image, fitted_pts):

        # Create copies for all segments and individual segments
        img_all = image.copy()
        img_ind = image.copy()

        # Initialize counter and dimensions
        h, w = image.shape[:2]
        masks = []
        np.random.seed(84)

        for key in fitted_pts.keys():
            
            
            # Set Contour Arrays
            x = np.array(fitted_pts[key]['x'])
            y = np.array(fitted_pts[key]['y'])

            # Stack x and y into Nx2 array: [[x0, y0], [x1, y1], ...]
            points = np.column_stack((x, y))
            contour = points.reshape((-1, 1, 2)).astype(np.int32)

            # Create empty mask for this contour
            mask = np.zeros((h, w), dtype=np.uint8)

            # Fill the contour on this mask
            cv2.drawContours(mask, [contour], contourIdx=-1, color=1, thickness=-1)

            # Add to mask list
            masks.append(mask)

        img_all = np.zeros((h, w, 3), dtype=np.uint8)
        img_ind = np.zeros((h, w, 3), dtype=np.uint8)

        # Generate random BGR colors
        num_masks = len(masks)
        colors = np.random.randint(0, 256, size=(num_masks, 3), dtype=np.uint8)

        for i, mask in enumerate(masks):
           # img_disp[mask == 1] = (color[0],color[1],color[2])
            img_all[mask == 1] = (color[2],color[1],color[0])
            img_ind[mask == 1] = colors[i]

        img_disp = image.copy()
            
        return img_disp, img_all, img_ind, masks
    
    # Function to resize the image
    def resize_image(img_all, masks):

        # Get New Sample Dimensions
        if Input['ReductionSize'] is not None:
            W = int(img_all.shape[0]*Input['ReductionSize'])
            L = int(img_all.shape[1]*Input['ReductionSize'])
        else:
            W = int(Input['W'])
            L = int(Input['L'])

        # Resize the image
        resized_img = cv2.resize(img_all, (L, W), interpolation=cv2.INTER_NEAREST)
        resized_masks = []
        for mask in masks:
            resized_masks.append(cv2.resize(mask, (L, W), interpolation=cv2.INTER_NEAREST))

        # Change black to white for visualization
        black_mask = np.all(resized_img == [0, 0, 0], axis=-1)
        resized_img[black_mask] = [255, 255, 255]

        return resized_img, resized_masks
    
    # Function to fill gaps
    def fill_gaps(resized_img, resized_masks, conv_color):

        # Create binary mask of white pixels
        white_mask = np.all(resized_img == [255, 255, 255], axis=-1).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(white_mask)

        # Set threshold
        area_threshold = 10

        # Find gaps
        for label in range(1, num_labels):
            component_area = np.sum(labels == label)
            if component_area < area_threshold:

                # Reset pixel values
                resized_img[labels == label] = conv_color

                # Find coordaintes where pixel values changed
                coords = np.where(labels == label)
                label_counts = []
                for mask in resized_masks:
                    # Extract patch around (y, x)
                    patch = mask[max([min(coords[0])-1,0]):min([max(coords[0])+2, mask.shape[0]]), max([min(coords[1])-1,0]):min([max(coords[1])+2, mask.shape[1]])]
                    
                    # Count how many pixels in the patch belong to this label
                    count = np.sum(patch)
                    label_counts.append(count)

                # Reset mask values
                if np.sum(label_counts) > 0:
                    lab = int(np.argmax(label_counts))
                    for i in range(len(coords[0])):
                        resized_masks[lab][coords[0][i]][coords[1][i]] = 1

        return resized_img, resized_masks
    
    # Function to remove nubs
    def remove_nubs(resized_img, resized_masks):
        # Get List of all patterns
        patterns = []
        opts = []
        coords = []
        # -- Right Nubs
        for i in range(Input['MaxNub']):
            pattern = np.ones(shape = (i+3, 2), dtype=np.uint8)
            pattern[0,1] = 0
            pattern[-1,1] = 0
            patterns.append(pattern)
            opts.append(1)

        # -- Left Nubs
        for i in range(Input['MaxNub']):
            pattern = np.ones(shape = (i+3, 2), dtype=np.uint8)
            pattern[0,0] = 0
            pattern[-1,0] = 0
            patterns.append(pattern)
            opts.append(2)

        # -- Bottom Nubs
        for i in range(Input['MaxNub']):
            pattern = np.ones(shape = (2, i+3), dtype=np.uint8)
            pattern[1,0] = 0
            pattern[1,-1] = 0
            patterns.append(pattern)
            opts.append(3)

        # -- Top Nubs
        for i in range(Input['MaxNub']):
            pattern = np.ones(shape = (2, i+3), dtype=np.uint8)
            pattern[0,0] = 0
            pattern[0,-1] = 0
            patterns.append(pattern)
            opts.append(4)

        for k, pattern in enumerate(patterns):
            
            # Get the size of the pattern
            ph, pw = pattern.shape

            # Perform cross-correlation
            for i, mask in enumerate(resized_masks):
                correlation = correlate2d(mask, pattern, mode='valid')

                # Check for exact matches by comparing with total number of 1s in the pattern
                match_score = np.sum(pattern)
                matches = (correlation == match_score)

                # To ensure perfect match (including 0s), compare the same for the inverted pattern
                pattern_inv = 1 - pattern
                correlation_inv = correlate2d(1 - mask, pattern_inv, mode='valid')
                match_score_inv = np.sum(pattern_inv)
                matches &= (correlation_inv == match_score_inv)

                # Get top-left coordinates of matches
                match_coords = np.argwhere(matches)

                if match_coords.shape[0] > 0:
                    for matchc in match_coords:
                        y = matchc[0]
                        x = matchc[1]

                        if opts[k] < 3:
                            for j in range(y+1,y+ph-1):
                                resized_masks[i][j][x+pw-opts[k]] = 0
                                resized_img[j][x+pw-opts[k]] = [255,255,255]
                                coords.append([j, x+pw-opts[k]])
                        else:
                            for j in range(x+1,x+pw-1):
                                resized_masks[i][y+ph - (opts[k]-2)][j] = 0
                                resized_img[y+ph - (opts[k]-2)][j] = [255,255,255]
                                coords.append([y+ph - (opts[k]-2), j])

        return resized_img, resized_masks
    
    # Function to remove corners
    def remove_corners(resized_img, resized_masks):
        # Get List of all patterns
        patterns = []
        opts = []
        coords = []
        # -- Top Right Corner
        pattern = np.ones(shape = (Input['MaxCorner'] + 1, Input['MaxCorner'] + 1), dtype=np.uint8)
        for i in range(Input['MaxCorner'] + 1):
            pattern[0, i] = 0
            pattern[i, -1] = 0
            opts.append(1)
        patterns.append(pattern)

        # -- Top Left Corner
        pattern = np.ones(shape = (Input['MaxCorner'] + 1, Input['MaxCorner'] + 1), dtype=np.uint8)
        for i in range(Input['MaxCorner'] + 1):
            pattern[0, i] = 0
            pattern[i, 0] = 0
            opts.append(2)
        patterns.append(pattern)

        # -- Bottom Left Corner
        pattern = np.ones(shape = (Input['MaxCorner'] + 1, Input['MaxCorner'] + 1), dtype=np.uint8)
        for i in range(Input['MaxCorner'] + 1):
            pattern[-1, i] = 0
            pattern[i, 0] = 0
            opts.append(3)
        patterns.append(pattern)

        # -- Bottom Right Corner
        pattern = np.ones(shape = (Input['MaxCorner'] + 1, Input['MaxCorner'] + 1), dtype=np.uint8)
        for i in range(Input['MaxCorner'] + 1):
            pattern[-1, i] = 0
            pattern[i, -1] = 0
            opts.append(4)
        patterns.append(pattern)
        

        for k, pattern in enumerate(patterns):
            
            # Get the size of the pattern
            ph, pw = pattern.shape

            # Perform cross-correlation
            for i, mask in enumerate(resized_masks):
                correlation = correlate2d(mask, pattern, mode='valid')

                # Check for exact matches by comparing with total number of 1s in the pattern
                match_score = np.sum(pattern)
                matches = (correlation == match_score)

                # To ensure perfect match (including 0s), compare the same for the inverted pattern
                pattern_inv = 1 - pattern
                correlation_inv = correlate2d(1 - mask, pattern_inv, mode='valid')
                match_score_inv = np.sum(pattern_inv)
                matches &= (correlation_inv == match_score_inv)

                # Get top-left coordinates of matches
                match_coords = np.argwhere(matches)

                if match_coords.shape[0] > 0:
                    for matchc in match_coords:
                        y = matchc[0]
                        x = matchc[1]

                        if opts[k] == 1:
                            yval = y+1
                            xval = x+pw-2

                        if opts[k] == 2:
                            yval = y+1
                            xval = x+1

                        if opts[k] == 3:
                            yval = y+ph-2
                            xval = x+1

                        if opts[k] == 3:
                            yval = y+ph-2
                            xval = x+pw-2

                        resized_masks[i][yval][xval] = 0
                        resized_img[yval][xval] = [255,255,255]
                        coords.append([yval, xval])

        return resized_img, resized_masks

    # Function to find touching pixels between masks
    def touching_pixels(masks, connectivity=2, dilation_iterations=1):
        """
        masks: list of (n, m) binary arrays, each representing a fiber
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity
        dilation_iterations: how many pixels to dilate each mask
        
        Returns: dict with keys (i, j) of touching fibers and values = array of touching coords
        """
        struct = ndimage.generate_binary_structure(2, connectivity)
        results = {}

        # Pre-dilate all masks
        dilated_masks = [ndimage.binary_dilation(mask, structure=struct, iterations=dilation_iterations) for mask in masks]

        # Compare every pair
        for i, dil_i in enumerate(dilated_masks):
            for j in range(i+1, len(masks)):
                overlap = dil_i & masks[j]
                if np.any(overlap):
                    coords = np.argwhere(overlap)
                    results[(i, j)] = coords

        return results
    
    # Function to calculate radii
    def calculate_radii(resized_masks):

        rads = []
        for mask in resized_masks:
            mask = mask.astype(bool)

            # Check if on boundary
            if (mask[0,:].any() or mask[-1,:].any() or
                mask[:,0].any() or mask[:,-1].any()):
                continue

            # Calculate radius from area
            area = mask.sum()
            radius = np.sqrt(area / np.pi)
            rads.append(radius)

        return rads
        

    # Load the Original Image
    image = Input['Image']

    # Set the color
    color = Input['Colors']

    # Convert Color to BGR Show original  mask
    conv_color = color[::-1]

    # Split the image
    output, markers = split_touching_circles(image, color)

    # Fit Circles to Fibers
    fitted, fitted_pts = fit_circles_from_markers(markers, image.shape)
    
    # Remove intersections
    fitted_pts = get_intersecting_boundaries(fitted_pts)

    # Generate Masks and New Images
    img_disp, img_all, img_ind, masks = generate_new_image(image, fitted_pts)

    # Resize the image and labels
    resized_img, resized_masks = resize_image(img_all, masks)

    # Fill Gaps
    resized_img, resized_masks = fill_gaps(resized_img, resized_masks, conv_color)

    # 9 .Remove Rouching Fibers
    if Input['TouchOption'] == True:
        results = touching_pixels(resized_masks)
        for res in results.keys():
            for pair in results[res]:
                resized_img[pair[0],pair[1],:] = [0, 0, 0]

                for mask in resized_masks:
                    mask[pair[0],pair[1]] = 0

    # Remove Nubs
    if Input['MaxNub'] > 0:
        resized_img, resized_masks = remove_nubs(resized_img, resized_masks)

    # Remove Corners
    if Input['MaxCorner'] > 0:
        resized_img, resized_masks = remove_corners(resized_img, resized_masks)
    
    # Caluclate Radiis
    rads = calculate_radii(resized_masks)

    # Create Output Mask
    mask = np.full_like(resized_img[:, :, 0], 2, dtype=np.uint8)
    for i, m in enumerate(resized_masks):   
        mask[m == 1] = 1  # Label fibers as one

    # Create Output
    # Calculate actual values
    out = {
            'VF':None,
           'NB':None,
           'NG':None,
           'R':None,
           'F':1,
           'M':2
           }
    
    # Set Dimensions
    nx = len(mask[0,:])
    ny = len(mask[:,0])

    # Calculate Volume Fraction
    out['VF'] = np.sum(mask == 1) / (nx * ny)

    # Calculate subcell dimensions
    out['NB'] = nx
    out['NG'] = ny
    
    # Calculate average radius
    if len(rads) > 0:
        out['R'] = np.mean(rads)

    # Return Data
    return mask, out