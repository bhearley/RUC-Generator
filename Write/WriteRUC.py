def WriteRUC(mask):
    ruc_data = '*RUC\n'
    ruc_data = ruc_data + f" MOD={202} ARCHID={99} \n"

    # Get Subcell Counts
    NB =  len(mask)
    NG =  len(mask[0]) 
    ruc_data = ruc_data + f" NB={NB} NG={NG} \n"

    # Write H
    ruc_data = ruc_data + ' H='
    for i in range(NB):
        ruc_data = ruc_data + '1'
        if i == NB - 1:
            ruc_data = ruc_data + '\n'
        else:
            ruc_data = ruc_data + ','

    # Write L
    ruc_data = ruc_data + ' L='
    for i in range(NG):
        ruc_data = ruc_data + '1'
        if i == NG - 1:
            ruc_data = ruc_data + '\n'
        else:
            ruc_data = ruc_data + ','

    for i in range(NB):
        ruc_data = ruc_data + " SM="
        for j in range(NG):
            ruc_data = ruc_data + f"{int(mask[i][j])}"
            if j != NG - 1:
                ruc_data = ruc_data + ","
        ruc_data = ruc_data + "\n"

    return ruc_data
