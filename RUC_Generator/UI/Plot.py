def Plot(mask, fib_color, mat_color, show_grid):
    """
    Generate the microstructure images for the streamlit UI

    Arguments:
        mask        2D array    integer array defining the microstructure
        fib_color   string      name of the fiber color
        mat_color   string      name of the matrix color
        show_grid   Boolean     indicator to show the grid or not

    Outputs:
        fig         figure      Plotly figure
    """
    # Import modules
    import plotly.graph_objects as go
    
    # Decide on grid spacing
    if show_grid:
        xgap = 0.5
        ygap = 0.5
    else:
        xgap = None
        ygap = None

    # Create Plotly figure
    fig = go.Figure(data=go.Heatmap(
            z=mask,
            colorscale=[[0, fib_color], 
                        [1, mat_color]],
            showscale=False,
            xgap=xgap,
            ygap=ygap
        ))

    # Layout tweaks 
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=False,
        height=350  
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        constrain="domain"
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        autorange='reversed',
        scaleanchor="x",
        constrain="domain"
    )

    return fig