def WriteCSV(mask):
    import pandas as pd
    from io import StringIO

    # Convert to CSV in memory
    csv_buffer = StringIO()
    pd.DataFrame(mask).to_csv(csv_buffer, index=False, header=False)
    csv_data = csv_buffer.getvalue()

    return csv_data