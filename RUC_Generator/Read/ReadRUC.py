def ReadRUC(content):
    """
    Read either a .txt or .mac file containing a RUC definition.

    Arguments:
        content str         content of the .mac/.txt file
    
    Outputs:
        mask    2D array    integer array defining the microstructure
        out     dict        dictionary of actual microstructure properties
    """
    
    # Import Modules
    import numpy as np

    # Initialize message
    msg = ""

    # Separate into lines
    lines = content.strip().splitlines()

    # Find *RUC section
    flag = 0
    end_line = len(lines)
    for i, line in enumerate(lines):
        if "*RUC" in line:
            start_line = i
            flag = 1
            continue
        if flag == 1:
            if "*" in line:
                end_line = i
                break

    if flag == 0:
        msg = msg + "No *RUC section found in the file."
        return None, None, msg
    
    # Check for 2D RUC
    is_2D = False
    for i in range(start_line, end_line):
        if "MOD=" in lines[i]:
            ndim = int(lines[i].split('MOD=')[1].split()[0][-1])
            if ndim == 2:
                is_2D = True
            break

    if not is_2D:
        msg = msg + "The RUC defined in the file is not 2D."
        return None, None, msg
    
    # Check for ARCHID = 99
    is_99 = False
    for i in range(start_line, end_line):
        if "ARCHID=" in lines[i]:
            archid = int(lines[i].split('ARCHID=')[1].split()[0])
            if archid == 99:
                is_99 = True
            break

    if not is_99:
        msg = msg + "The RUC defined in the file is not ARCHID = 99."
        return None, None, msg

    # Get subcells dimensions
    H = ''
    flag_h = 0
    for i in range(start_line, end_line):
        if "H=" in lines[i]:
            H = H + lines[i].split('H=')[1]
            flag_h = 1
            continue
        if flag_h == 1:
            if "=" in lines[i]:
                break
            else:
                H = H + lines[i]
    H = H.replace('&','')
    H = H.replace('\n','')
    H = H.replace(' ','')
    H = [float(x) for x in H.split(",") if x.strip() != ""]
    NB = len(H)

    L = ''
    flag_l = 0
    for i in range(start_line, end_line):
        if "L=" in lines[i]:
            L = L + lines[i].split('L=')[1]
            flag_l = 1
            continue
        if flag_l == 1:
            if "=" in lines[i]:
                break
            else:
                L = L + lines[i]
    L = L.replace('&','')
    L = L.replace('\n','')
    L = L.replace(' ','')
    L = [float(x) for x in L.split(",") if x.strip() != ""]
    NG = len(L)

    # Get the subcell values
    mask = np.zeros(shape=(NB,NG))

    for j in range(NB):
        SM = ''
        flag_m = 0
        for i in range(start_line, end_line):
            if flag_m == 0:
                if "SM=" in lines[i]:
                    SM = SM + lines[i].split('SM=')[1]
                    flag_m = 1
                    continue
            if flag_m == 1:
                if "SM=" in lines[i]:
                    start_line = i
                    break
                else:
                    SM = SM + lines[i]
        SM = SM.replace('&','')
        SM = SM.replace('\n','')
        SM = SM.replace(' ','')
        SM = [int(x) for x in SM.split(",") if x.strip() != ""]
        mask[j,:] = SM
    mask = mask.astype(int)

    # Calculate actual values
    out = {
            'VF':None,
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

    # Calculate subcell dimensions
    out['NB'] = nx
    out['NG'] = ny

    return mask, out, msg