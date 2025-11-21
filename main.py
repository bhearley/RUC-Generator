# Import Modules
import cv2
import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import zipfile

# Import Functions
from Hexagonal.Hex1 import Hex1
from Hexagonal.Hex2 import Hex2
from Hexagonal.Hex3 import Hex3
from Square.Square1 import Square1
from Square.Square2 import Square2
from Square.Square3 import Square3
from Write.WriteCSV import WriteCSV
from Write.WriteRUC import WriteRUC
from Read.ReadCSV import ReadCSV
from Read.ReadRUC import ReadRUC
from Random.RandomRVE import RandomRVE
from Segmented.Segmented import Segmented

# Set the page configuration
st.set_page_config(layout="wide")

# Create the title
st.title("2D NASMAT RUC Generator")

# Create instructions
st.markdown('The 2D NASMAT RUC Generator is used to generate and visualize microstructres and corresponding *RUC files compatible with the NASA Mulitscale Analysis Tool (NASMAT). \n \n')

# Create tabs
tab_ord, tab_rand, tab_img, tab_viz = st.tabs(["Ordered", "Random", "From Image", "Visualizer"])

# Ordered Microstructure Generation
with tab_ord:
    # Create Header
    st.markdown("## Ordered RUC Generator")
    st.markdown('Generate a Repeating Unit Cell (RUC) microstructure based on user-defined parameters. Select the microstructure type and input parameters to visualize and download the RUC data.')

    # Initialize the function
    func = None

    # Create columns for microstructure and definition selection
    col_ord_1, col_ord_2 = st.columns([1, 1])

    # Create input for microstructure type
    with col_ord_1:
        micro_opt_ord = st.selectbox(
                            "Select a microstructure:",
                                [
                                    "Hexagonal", 
                                    "Square",
                                ],
                            key = 'micro_opt_ord',
                        )

    # Select definition option
    if micro_opt_ord == "Hexagonal":

        # -- Create default values
        def_vals = {
                'VF':[1, 'float', 0.001, 0., math.pi / (2*math.sqrt(3)), 0.6],
                'R':[2, 'float', 0.001, 0., None, 10.],
                'NB':[1, 'int', 1, 1, None, 10],
                'NG':[2, 'int', 1, 1, None, 10],
                'F':[1,'int', 1, 1, None, 1],
                'M':[2,'int', 1, 1, None, 2],
                }

        # -- Create defintion list
        def_list = {"Volume Fraction & Subcell Dimensions":{
                                                            'Inputs':['VF','NB','F','M'],
                                                            'Function':Hex1
                                                            }, 
                    "Volume Fraction & Radius":{
                                                'Inputs':['VF','R','F','M'],
                                                'Function':Hex2
                                                }, 
                    "Radius & Subcell Dimensions":{
                                                    'Inputs':['R','NB','F','M'],
                                                    'Function':Hex3
                                                    }, 
                    }
        
    elif micro_opt_ord == "Square":
        
        # -- Create default values
        def_vals = {
                'VF':[1, 'float', 0.001, 0., math.pi / 4, 0.6],
                'R':[2, 'float', 0.001, 0., None, 10.],
                'NB':[1, 'int', 1, 1, None, 10],
                'NG':[2, 'int', 1, 1, None, 10],
                'F':[1,'int', 1, 1, None, 1],
                'M':[2,'int', 1, 1, None, 2],
                }

        # -- Create defintion list
        def_list = {"Volume Fraction & Subcell Dimensions":{
                                                            'Inputs':['VF','NB','F','M'],
                                                            'Function':Square1
                                                            }, 
                    "Volume Fraction & Radius":{
                                                'Inputs':['VF','R','F','M'],
                                                'Function':Square2
                                                }, 
                    "Radius & Subcell Dimensions":{
                                                    'Inputs':['R','NB','F','M'],
                                                    'Function':Square3
                                                    }, 
                    }
        
    else:
        def_list = {}

    # Create definition selection
    with col_ord_2:
        def_opt_ord = st.selectbox(
                        "Select an input type:", 
                        list(def_list.keys()),
                        key = 'def_opt_ord',
                        )

    # Create input selection area
    st.markdown('''---''')

    # Create user inputs
    if def_opt_ord:

        # Get the function and initalize values
        func = def_list[def_opt_ord]['Function']
        values = {}

        # Separeate numeric inputs into two columns
        col_ord_3, col_ord_4 = st.columns([1, 1])

        # Create numeric inputs
        for key in def_vals.keys():

            # Determine column
            colnum = col_ord_3 if def_vals[key][0] == 1 else col_ord_4

            with colnum:

                # Get min, max, step, default
                min_v = def_vals[key][3]
                max_v = def_vals[key][4]
                step = def_vals[key][2]
                default = def_vals[key][5]

                # Set widget key
                widget_key = f"num_input_{key}_ord"

                # Populate Values
                if key in def_list[def_opt_ord]['Inputs']:
                    # ACTIVE input

                    # Restore previous or use default
                    if widget_key in st.session_state:
                        val = st.session_state[widget_key]

                        # Clamp if needed
                        if min_v is not None and val < min_v:
                            val = min_v
                        if max_v is not None and val > max_v:
                            val = max_v
                    else:
                        val = default

                    # Render input (this updates session_state automatically)
                    values[key] = st.number_input(
                        key,
                        key=widget_key,
                        value=val,
                        step=step,
                        min_value=min_v,
                        max_value=max_v,
                    )

                else:
                    # INACTIVE input → remove its session state
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]

                    # Use a disabled text field that LOOKS empty
                    st.text_input(key, value="", disabled=True, key = key+"_ord")

                    values[key] = None
                            

    # Generate and display the RUC
    if func is not None:

        # -- Create columns for organization
        col_ord_5, col_ord_6, col_ord_7, col_ord_8, col_ord_9 = st.columns([1, 1, 1, 1, 7])

        # -- Create the generate RUC button
        with col_ord_5:
            st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
            generate_clicked_ord = st.button("Generate RUC", key='gen_button_ord')
            st.write("")

        # -- Create the gridline checkbox
        with col_ord_6:
            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
            show_grid_ord = st.checkbox("Show Grid Lines", value=True, key='grid_check_ord')
            st.write("")

        # -- Create fiber and matrix color selectors
        with col_ord_7:
            if 'fiber_color_ord' not in st.session_state:
                st.session_state['fiber_color_ord'] = 'blue'
            fib_color = st.selectbox(
                    "Fiber Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'fiber_color_ord'
                )
            st.write("")
            
        with col_ord_8:
            if 'matrix_color_ord' not in st.session_state:
                st.session_state['matrix_color_ord'] = 'red'
            mat_color = st.selectbox(
                    "Matrix Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'matrix_color_ord'
                )
            st.write("")

        # If generate is clicked, run function and save mask in session_state
        if generate_clicked_ord:
            func_values = {}
            flag = 0
            for key in values.keys():
                if key in def_list[def_opt_ord]['Inputs']:
                    func_values[key] = values[key]
            st.session_state['mask_ord'] = func(**func_values)

        # Only plot if we have a mask
        if 'mask_ord' in st.session_state:
            mask, out = st.session_state['mask_ord']

            # Decide on grid spacing
            if show_grid_ord:
                xgap = 0.5
                ygap = 0.5
            else:
                xgap = None
                ygap = None

            # Create Plotly figure
            fig = go.Figure(data=go.Heatmap(
                    z=mask,
                    colorscale=[[0, st.session_state['fiber_color_ord']], 
                                [1, st.session_state['matrix_color_ord']]],
                    showscale=False,
                    xgap=xgap,
                    ygap=ygap
                ))

            # Layout tweaks (fixed)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=False,
                height=350   # tweak to fit your grid perfectly
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

            # Create columns for visualalization and data
            col_ord_10, col_ord_11, col_ord_12 = st.columns([1.1, 1.15, 3.85])

            # Display the microstruture
            with col_ord_10:
                st.plotly_chart(fig, width='content', key = 'plot_ord')

            # Create table with actual microstructure properties
            with col_ord_11:
                data = {'Property':['VF', 'R', 'NB', 'NG'],
                        'Value':[out['VF'], out['R'], out['NB'], out['NG']]}
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_ord', hide_index=True) 

            # Create Files
            csv_data = WriteCSV(mask)
            ruc_data = WriteRUC(mask)

            # Create columns for downloading data
            col_ord_13, col_ord_14, col_ord_15 = st.columns([1, 1, 9])

            # Download to CSV
            with col_ord_13:
                st.download_button(
                    label="Download  CSV",
                    data=csv_data,
                    file_name="ruc.csv",
                    mime="text/csv",
                    key="download_csv_ord"
                )

            # Download for *RUC
            with col_ord_14:
                st.download_button(
                label="Download *RUC File",
                data=ruc_data,
                file_name="ruc_data.txt",
                mime="text/plain",
                key="download_ruc_ord"
            )

# Random Microstructure Generation   
with tab_rand:
    # Create Header
    st.markdown("## Random RVE Generator")
    st.markdown('Generate a random Representative Volume Element (RVE) microstructure based on user-defined parameters. Select the input parameters to visualize and download the RUC data.')

    # Initialize the function
    func = None
    
    # Create columns for microstructure and definition selection
    col_rand_1, col_rand_2 = st.columns([1, 1])

    # Create input for microstructure type
    num_gen_rand = st.number_input(
                        'Number of Random Generations',
                        key='num_gen_rand',
                        value=1,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
    
    with col_rand_1:
        vf_rand = st.number_input(
                        'Volume Fraction',
                        key='vf_rand',
                        value=0.5,
                        step=0.01,
                        min_value=0.01,
                        max_value=0.65,
                        )
        num_NB_rand = st.number_input(
                        'NB',
                        key='num_NB_rand',
                        value=100,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
        min_gap_rand = st.number_input(
                        'Minimum Gap Between Fibers',
                        key='min_gap_rand',
                        value=3,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
        
    # Create definition selection
    with col_rand_2:
        num_fib_rand = st.number_input(
                        'Number of Fibers',
                        key='num_fib_rand',
                        value=16,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
        num_NG_rand = st.number_input(
                        'NG',
                        key='num_NG_rand',
                        value=100,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
        max_iter_rand = st.number_input(
                        'Maximum Radius Resize Iterations',
                        key='max_iter_rand',
                        value=20,
                        step=1,
                        min_value=1,
                        max_value=None,
                        )
        
    generate_random = st.button("Generate RVEs", key = 'gen_button_rand')

    if generate_random:
        # Generate multiple RVEs
        masks = []
        for i in range(int(num_gen_rand)):
            mask, out = RandomRVE(
                                    W = int(num_NB_rand),
                                    H = int(num_NG_rand),
                                    N_fibers = int(num_fib_rand),
                                    VF = vf_rand,
                                    min_gap_subcells = int(min_gap_rand),
                                    tol = 1e-3,
                                    max_iter_radius = int(max_iter_rand)
                                )
            masks.append((f'RVE {i+1}',mask, out))

        # Save generated RVEs to session state
        st.session_state['masks_rand'] = masks
        st.session_state['select_rve_rand'] = masks[0][0]  # default selection

    # Only display RVE selection if we have generated RVEs
    if 'masks_rand' in st.session_state:
        masks = st.session_state['masks_rand']

        # Organize into columns
        col_rand_3, col_rand_4, col_rand_5, col_rand_6, col_rand_7 = st.columns([1, 1, 1, 1, 6])

        with col_rand_3:
            # Create Select Box to choose RVE
            rve_names = [name for name,_,_ in masks]
            selected_rve = st.selectbox(
                            "Select an RVE to visualize:", 
                            rve_names,
                            key = 'select_rve_rand',
                            )
            st.write("")
            
        with col_rand_4:
            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)

            # Gridline checkbox
            show_grid_rand = st.checkbox("Show Grid Lines", value=True, key='grid_check_rand')
            st.write("")

        # -- Create fiber and matrix color selectors
        with col_rand_5:
            if 'fiber_color_rand' not in st.session_state:
                st.session_state['fiber_color_rand'] = 'blue'
            fib_color = st.selectbox(
                    "Fiber Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'fiber_color_rand'
                )
            st.write("")
            
        with col_rand_6:
            if 'matrix_color_rand' not in st.session_state:
                st.session_state['matrix_color_rand'] = 'red'
            mat_color = st.selectbox(
                    "Matrix Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'matrix_color_rand'
                )
            st.write("")
            
        if 'select_rve_rand' in st.session_state:
            # Get the selected mask
            for name, mask, out in masks:
                if name == st.session_state['select_rve_rand']:
                    selected_mask = mask
                    selected_out = out
                    break

            # Decide on grid spacing
            if show_grid_rand:
                xgap = 0.5
                ygap = 0.5
            else:
                xgap = None
                ygap = None

            fig = go.Figure(data=go.Heatmap(
                    z=selected_mask,
                    colorscale=[[0, st.session_state['fiber_color_rand']], 
                                [1, st.session_state['matrix_color_rand']]],
                    showscale=False,
                    xgap=xgap,
                    ygap=ygap
                ))

            # Layout tweaks (fixed)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=False,
                height=350   # tweak to fit your grid perfectly
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

            # Create columns for visualalization and data
            col_rand_8, col_rand_9, col_rand_10 = st.columns([1.25, 1.25, 3.75])

            # Display the microstruture
            with col_rand_8:
                st.plotly_chart(fig, width='content', key='plot_rand')

            # Create table with actual microstructure properties
            with col_rand_9:
                data = {'Property':['VF', 'R', 'NB', 'NG'],
                        'Value':[selected_out['VF'], selected_out['R'], selected_out['NB'], selected_out['NG']]}
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_rand', hide_index=True)

            # Create Files
            csv_data = WriteCSV(mask)
            ruc_data = WriteRUC(mask)

            # Create columns for downloading data
            col_rand_10, col_rand_11, col_rand_12 = st.columns([1, 1.2, 9])

            # Generate CSV Files
            def generate_zip_with_masks_csv(masks):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for name, mask, _ in masks:
                        csv_content = WriteCSV(mask)  # Get CSV string from mask
                        zip_file.writestr(f"{name}.csv", csv_content)
                zip_buffer.seek(0)
                return zip_buffer

            zip_bytes_csv = generate_zip_with_masks_csv(st.session_state['masks_rand'])
            

            # Download to CSV
            with col_rand_10:
                    st.download_button(
                    label="Download All to CSV",
                    data=zip_bytes_csv,
                    file_name="RVEs.zip",
                    mime="application/zip",
                    key = "download_csv_rand"
                    )

            # Generate CSV Files
            def generate_zip_with_masks_ruc(masks):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for name, mask, _ in masks:
                        txt_content = WriteRUC(mask)  # Get .txt content from mask
                        zip_file.writestr(f"{name}.txt", txt_content)
                zip_buffer.seek(0)
                return zip_buffer

            zip_bytes_ruc = generate_zip_with_masks_ruc(st.session_state['masks_rand'])

            # Download for *RUC
            with col_rand_11:
                st.download_button(
                label="Download All *RUC Files",
                data=zip_bytes_ruc,
                file_name="RVEs_MAC.zip",
                mime="application/zip",
                key = "download_ruc_rand"
                )
                    
with tab_img:
    st.markdown("## Generate RUC from Segmented Image")
    st.markdown("Generate a RUC microstructure from a segmented image. Upload a segmented image and specify parameters to visualize and download the RUC data.")

    # Allow file upload
    uploaded_file = st.file_uploader("Choose a file", type=["png","jpg"], key = 'file_upload_img')

    # Display the image
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

        col_img_1, col_img_2, col_img_3 = st.columns([1.2, 1, 1])

        with col_img_1:
            st.image(img, channels="BGR")

            blue_mask  = np.all(img == [255, 0, 0], axis=2)
            green_mask = np.all(img == [0, 255, 0], axis=2)
            red_mask   = np.all(img == [0, 0, 255], axis=2)

            colors = []
            if np.any(blue_mask):
                colors.append('Blue')
            if np.any(green_mask):
                colors.append('Green')
            if np.any(red_mask):
                colors.append('Red')

        def on_red_size_change():
            st.session_state.w_img = None
            st.session_state.h_img = None

        def on_w_img_change():
            st.session_state.red_size_img = None
            # If h_img is None or "other", set default of 100
            if st.session_state.h_img is None:
                st.session_state.h_img = 100

        def on_h_img_change():
            st.session_state.red_size_img = None
            # If w_img is None or "other", set default of 100
            if st.session_state.w_img is None:
                st.session_state.w_img = 100

        if 'w_img' not in st.session_state:
            st.session_state.w_img = None
        if 'h_img' not in st.session_state:
            st.session_state.h_img = None
        if 'red_size_img' not in st.session_state:
            st.session_state.red_size_img = 0.2

        with col_img_2:
            color = st.selectbox(
                "Select Fiber Color in Image",
                colors,
                key='color_select_img'
            )

            w_img = st.number_input(
                'RUC Width (subcells)',
                key='w_img',
                step=1,
                min_value=1,
                max_value=None,
                on_change=on_w_img_change
            )

            nub_img = st.number_input(
                'Max Nub Length (pixels)',
                key='nub_img',
                step=1,
                min_value=1,
                max_value=5,
                value = 1,
            )

            touch_img = st.checkbox("Remove Touching Fiber Subcells", value=True, key='touch_check_img')

        with col_img_3:
            red_size = st.number_input(
                'Reduction Size',
                key='red_size_img',
                step=0.01,
                min_value=0.01,
                max_value=1.,
                on_change=on_red_size_change
            )

            h_img = st.number_input(
                'RUC Height (subcells)',
                key='h_img',
                step=1,
                min_value=1,
                max_value=None,
                on_change=on_h_img_change
            )

            corner_img = st.number_input(
                'Max Corner Length (pixels)',
                key='corner_img',
                step=1,
                min_value=1,
                max_value=5,
                value = 1,
            )

        # -- Create columns for organization
        col_img_4, col_img_5, col_img_6, col_img_7, col_img_8 = st.columns([1, 1, 1, 1, 7])

        # -- Create the generate RUC button
        with col_img_4:
            st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
            generate_clicked_img = st.button("Generate RUC", key = 'gen_button_img')
            st.write("")

        # -- Create the gridline checkbox
        with col_img_5:
            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
            show_grid_img = st.checkbox("Show Grid Lines", value=True, key='grid_check_img')
            st.write("")

        # -- Create fiber and matrix color selectors
        with col_img_6:
            if 'fiber_color_img' not in st.session_state:
                st.session_state['fiber_color_img'] = 'blue'
            fib_color = st.selectbox(
                    "Fiber Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'fiber_color_img'
                )
            st.write("")
            
        with col_img_7:
            if 'matrix_color_img' not in st.session_state:
                st.session_state['matrix_color_img'] = 'red'
            mat_color = st.selectbox(
                    "Matrix Color",
                    ["white", "black", "red", "green", "blue", "yellow", "purple"],
                    key = 'matrix_color_img'
                )
            st.write("")

        # If generate is clicked, run function and save mask in session_state
        if generate_clicked_img:
            if color == 'Blue':
                img_color = [0, 0, 255]
            if color == 'Green':
                img_color = [0, 255, 0]
            if color == 'Red':
                img_color = [255, 0, 0]

            Input = {
                'Image':img,
                'ReductionSize':red_size,
                'W':w_img,
                'L':h_img,
                'Colors':img_color,
                'MaxNub':nub_img,
                'MaxCorner':corner_img,
                'TouchOption': touch_img,
                }
            st.session_state['mask_img'] = Segmented(Input)

        # Only plot if we have a mask
        if 'mask_img' in st.session_state:
            mask, out = st.session_state['mask_img']

            # Decide on grid spacing
            if show_grid_img:
                xgap = 0.5
                ygap = 0.5
            else:
                xgap = None
                ygap = None

            # Create Plotly figure
            fig = go.Figure(data=go.Heatmap(
                    z=mask,
                    colorscale=[[0, st.session_state['fiber_color_img']], 
                                [1, st.session_state['matrix_color_img']]],
                    showscale=False,
                    xgap=xgap,
                    ygap=ygap
                ))

            # Layout tweaks (fixed)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=False,
                height=350   # tweak to fit your grid perfectly
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

            # Create columns for visualalization and data
            col_img_9, col_img_10, col_img_11 = st.columns([1.1, 1.15, 3.85])

            # Display the microstruture
            with col_img_9:
                st.plotly_chart(fig, width='content', key = 'plot_img')

            # Create table with actual microstructure properties
            with col_img_10:
                data = {'Property':['VF', 'R', 'NB', 'NG'],
                        'Value':[out['VF'], out['R'], out['NB'], out['NG']]}
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_img', hide_index=True) 

            # Create Files
            csv_data = WriteCSV(mask)
            ruc_data = WriteRUC(mask)

            # Create columns for downloading data
            col_img_12, col_img_13, col_img_14 = st.columns([1, 1, 9])

            # Download to CSV
            with col_img_12:
                st.download_button(
                    label="Download  CSV",
                    data=csv_data,
                    file_name="ruc.csv",
                    mime="text/csv",
                    key="download_csv_img"
                )

            # Download for *RUC
            with col_img_13:
                st.download_button(
                label="Download *RUC File",
                data=ruc_data,
                file_name="ruc_data.txt",
                mime="text/plain",
                key="download_ruc_img"
            )
                
# RUC Visualizer
with tab_viz:

    # Create header
    st.markdown("## RUC Visualizer")
    st.markdown("Upload a CSV or *RUC file to visualize the microstructure.")

    # Set flag
    flag = 0

    # Allow file upload
    uploaded_file = st.file_uploader("Choose a file", type=["txt","mac","csv"], key = 'file_upload_viz')

    # Read Data
    if uploaded_file is not None:
        # Read file as string (for text files)
        content = uploaded_file.read().decode("utf-8")

        # Read a csv file
        if uploaded_file.name.endswith('.csv'):
            try:
                mask, out = ReadCSV(content)
                st.session_state['mask_viz'] = mask
                flag = 1
            except:
                st.error("Error reading CSV file. Please ensure it is formatted correctly.")

        else:
                mask, out, msg = ReadRUC(content)
                if msg != "":
                    st.error(msg)
                else:
                    st.session_state['mask_viz'] = mask
                    flag = 1


        # Display RUC
        if flag == 1:
            # -- Create columns for organization
            col_viz_1, col_viz_2, col_viz_3, col_viz_4 = st.columns([1, 1, 1, 8])

            # -- Create the gridline checkbox
            with col_viz_1:
                st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
                show_grid_viz = st.checkbox("Show Grid Lines", value=True, key='grid_check_viz')
                st.write("")

            # -- Create fiber and matrix color selectors
            with col_viz_2:
                if 'fiber_color_viz' not in st.session_state:
                    st.session_state['fiber_color_viz'] = 'blue'
                fib_color = st.selectbox(
                        "Fiber Color",
                        ["white", "black", "red", "green", "blue", "yellow", "purple"],
                        key = 'fiber_color_viz'
                    )
                st.write("")
                
            with col_viz_3:
                if 'matrix_color_viz' not in st.session_state:
                    st.session_state['matrix_color_viz'] = 'red'
                mat_color = st.selectbox(
                        "Matrix Color",
                        ["white", "black", "red", "green", "blue", "yellow", "purple"],
                        key = 'matrix_color_viz'
                    )
                st.write("")

            # Only plot if we have a mask
            if 'mask_viz' in st.session_state:
                mask = st.session_state['mask_viz']

                # Decide on grid spacing
                if show_grid_viz:
                    xgap = 0.5
                    ygap = 0.5
                else:
                    xgap = None
                    ygap = None

                fig = go.Figure(data=go.Heatmap(
                    z=mask,
                    colorscale=[[0, st.session_state['fiber_color_viz']], 
                                [1, st.session_state['matrix_color_viz']]],
                    showscale=False,
                    xgap=xgap,
                    ygap=ygap
                ))

                # Layout tweaks (fixed)
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    autosize=False,
                    height=350   # tweak to fit your grid perfectly
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

                # Create columns for visualalization and data
                col_viz_5, col_viz_6, col_viz_7 = st.columns([1.1, 1, 4])

                # Display the microstruture
                with col_viz_5:
                    st.plotly_chart(fig, width='content', key='plot_viz')

                # Create table with actual microstructure properties
                with col_viz_6:
                    data = {'Property':['VF', 'NB', 'NG'],
                            'Value':[out['VF'], out['NB'], out['NG']]}
                    df = pd.DataFrame(data)
                    df = df.reset_index(drop=True)
                    st.markdown("")
                    st.dataframe(df, key = 'out_table_viz', hide_index=True) 

                # Create Files
                csv_data = WriteCSV(mask)
                ruc_data = WriteRUC(mask)

                # Create columns for downloading data
                col_viz_8, col_viz_9, col_viz_10 = st.columns([1, 1, 9])

                # Download to CSV
                with col_viz_8:
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="ruc.csv",
                        mime="text/csv",
                        key = "download_csv_viz"
                    )

                # Download for *RUC
                with col_viz_9:
                    st.download_button(
                    label="Download *RUC File",
                    data=ruc_data,
                    file_name="ruc_data.txt",
                    mime="text/plain",
                    key="download_ruc_viz"
                )



