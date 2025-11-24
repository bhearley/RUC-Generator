def WriteCSV(mask):
    """
    Generate a CSV of the mask.

    Arguments:
        mask        2D array    integer array defining the microstructure

    Outputs:
        csv_data    str         str for streamlit to write to csv
    """

    # Import Modules
    import pandas as pd
    from io import StringIO

    # Convert to CSV in memory
    csv_buffer = StringIO()
    pd.DataFrame(mask).to_csv(csv_buffer, index=False, header=False)
    csv_data = csv_buffer.getvalue()

    return csv_data