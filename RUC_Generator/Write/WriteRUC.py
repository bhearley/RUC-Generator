def WriteRUC(mask, max_len=400):
    """
    Generate the *RUC portion of a NASMAT input deck ensuring each line
    does not exceed max_len characters. Lines longer than max_len are
    split using '&' continuation.

    Arguments:
        mask        2D array    integer array defining the microstructure
        max_len     int         maximum allowed line length before splitting

    Outputs:
        ruc_data    str         formatted input section
    """

    def append_safe(ruc, text, max_len=400):
        """Append text to ruc safely, inserting '&' and newlines if needed."""
        lines = ruc.split("\n")
        current = lines[-1]  # last line being built

        for ch in text:
            if len(current) >= max_len:
                # Break line with continuation
                lines[-1] = current + "&"
                lines.append("")  # Start new physical line
                current = ""

            current += ch

        lines[-1] = current
        return "\n".join(lines)

    # Begin writing the text data
    ruc = "*RUC\n"
    ruc += f" MOD={202} ARCHID={99}\n"

    # Get Subcell Counts
    NB = len(mask)
    NG = len(mask[0])
    ruc += f" NB={NB} NG={NG}\n"

    # Write H
    ruc += " H="
    for i in range(NB):
        ruc = append_safe(ruc, "1" + ("," if i != NB - 1 else "\n"), max_len)

    # Write L
    ruc += " L="
    for i in range(NG):
        ruc = append_safe(ruc, "1" + ("," if i != NG - 1 else "\n"), max_len)

    # Write SM rows
    for i in range(NB):
        ruc += " SM="
        for j in range(NG):
            entry = f"{int(mask[i][j])}"
            if j != NG - 1:
                entry += ","
            else:
                entry += "\n"
            ruc = append_safe(ruc, entry, max_len)

    return ruc
