#------------------------------------------------------------------------------------------------------------------------------------------
#
#   2D NASMAT RUC GENERATOR
#   
#   The 2D NASMAT RUC Generator is used to generate and visualize microstructres and corresponding *RUC files compatible with the NASA 
#   Mulitscale Analysis Tool (NASMAT). It allows users to generate ordered RUCs (square or hexagonal pack), random RUCs (with and 
#   without periodicity), RUCs from a segmented microscopy image (assuming circular fibers), or visualize an RUC from a *.mac input file.
#   Additionally, from either a segmented or visualized micorstructure, a statistically equivlaent, periodic RUC can be generated using
#   soft body dynamics to minimize the error in local fiber volume fraction using Bayesian optimization.
#
#   Brandon L. Hearley (LMS)
#   brandon.l.hearley@nasa.gov 
#   v1.0    11/24/25
#
#------------------------------------------------------------------------------------------------------------------------------------------

# Import Libraries
import cv2
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageColor
import streamlit as st
import zipfile

# Import Functions
from RUC_Generator.Random.RandomCharacterization import RandomCharacterization
from RUC_Generator.Read.ReadCSV import ReadCSV
from RUC_Generator.Read.ReadRUC import ReadRUC
from RUC_Generator.Segmented.Segmented import Segmented
from RUC_Generator.Write.WriteCSV import WriteCSV
from RUC_Generator.Write.WriteRUC import WriteRUC
from RUC_Generator.UI.Plot import Plot
from RUC_Generator.UI.UI_Definitions import UI_Definitions

#------------------------------------------------------------------------------------------------------------------------------------------
#   GENERAL SETUP
#   General settings for the entire UI.
#------------------------------------------------------------------------------------------------------------------------------------------ 

# Set the page configuration
st.set_page_config(layout="wide")

# Create the title
st.title("2D NASMAT RUC Generator")

# Create description
st.markdown("""The 2D NASMAT RUC Generator is used to generate and visualize microstructres and corresponding *RUC files compatible with 
            the NASA Mulitscale Analysis Tool (NASMAT).""")

# Create tabs
tab_ord, tab_rand, tab_img, tab_viz = st.tabs(["Ordered", "Random", "From Image", "Visualizer"])

#------------------------------------------------------------------------------------------------------------------------------------------
#   ORDERED MICROSTRUCTURE
#   Generate a periodic square or hexagonal pack microstructure.
#------------------------------------------------------------------------------------------------------------------------------------------ 

# Ordered Microstructure Generation
with tab_ord:

    # Create Header
    st.markdown("## Ordered RUC Generator")

    # Create description
    st.markdown("""Generate a Repeating Unit Cell (RUC) microstructure based on user-defined parameters. Select the microstructure 
                type and input parameters to visualize and download the RUC data.""")

    # Initialize the function
    func_ord = None

    # Create reset function
    def reset_ord_plot():
        if 'mask_ord' in st.session_state:
            del st.session_state['mask_ord']

    # Create columns for microstructure and definition selection
    col_ord_def_1, col_ord_def_2 = st.columns([1, 1])

    # Get list of available ordered microstructures
    ordered_list = UI_Definitions('OrderedList')

    # Create input for microstructure type
    with col_ord_def_1:

        # Get the ordered list
        ordered_list = UI_Definitions("OrderedList")

        # Create the input
        micro_opt_ord = st.selectbox(
                                    "Select a microstructure:",
                                    ordered_list,
                                    key = 'micro_opt_ord',
                                    on_change=reset_ord_plot
                                    )
        
        # Create option for interface
        st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
        interface_ord = st.checkbox('Include Interface', key = 'check_interface_ord', on_change=reset_ord_plot)

    # Hexagonal Pack Definition
    if micro_opt_ord == "Hexagonal":
        def_vals_ord, def_list_ord = UI_Definitions('Hexagonal')
        
    # Square Pack Definition
    elif micro_opt_ord == "Square":
        def_vals_ord, def_list_ord = UI_Definitions('Square')
        
    # Empty dictionary for option not selected    
    else:
        def_list_ord = {}

    # Create definition selection
    with col_ord_def_2:
        def_opt_ord = st.selectbox(
                                "Select an input type:", 
                                list(def_list_ord.keys()),
                                key = 'def_opt_ord',
                                on_change=reset_ord_plot
                                )
        
        # Create inputs for interface
        if interface_ord:

            # Get definition selection
            interface_opts, int_vals_ord_all = UI_Definitions('InterfaceOptions')

            # Create the input
            interface_opt_ord = st.selectbox(
                                        "Interface Definition:",
                                        interface_opts,
                                        key = 'interface_opt_ord',
                                        on_change=reset_ord_plot
                                        )
            
    # Separate user inputs
    st.markdown('''---''')

    # Create user inputs
    if def_opt_ord:

        # Get the function and initalize values
        func_ord = def_list_ord[def_opt_ord]['Function']
        values = {}

        # Separeate numeric inputs into two columns
        col_ord_inp_1, col_ord_inp_2 = st.columns([1, 1])

        # Create numeric inputs
        for key in def_vals_ord.keys():

            # Determine column
            colnum = col_ord_inp_1 if def_vals_ord[key][0] == 1 else col_ord_inp_2

            with colnum:

                # Get min, max, step, default, and display name
                step        = def_vals_ord[key][2]
                min_v       = def_vals_ord[key][3]
                max_v       = def_vals_ord[key][4]
                default     = def_vals_ord[key][5]
                disp_name   = def_vals_ord[key][6]
                frmt        = def_vals_ord[key][7]

                # Set widget key
                widget_key = f"num_input_{key}_ord"

                # Active Input
                if key in def_list_ord[def_opt_ord]['Inputs']:

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

                    # Render input 
                    values[key] = st.number_input(
                                                disp_name,
                                                key=widget_key,
                                                value=val,
                                                step=step,
                                                min_value=min_v,
                                                max_value=max_v,
                                                on_change=reset_ord_plot,
                                                format = frmt
                                                )

                # Inactive Input
                else:
                
                    # Remove its session state
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]

                    # Use a disabled text field that looks empty
                    st.text_input(disp_name, value="", disabled=True, key = key+"_ord")

                    # Reset value to None
                    values[key] = None

    # Create user inputs
    if interface_ord:

        # Get the correct inputs
        int_vals_ord = int_vals_ord_all[st.session_state['interface_opt_ord']]

        # Create numeric inputs
        for key in int_vals_ord.keys():

            # Determine column
            colnum = col_ord_inp_1 if int_vals_ord[key][0] == 1 else col_ord_inp_2

            with colnum:

                # Get min, max, step, default, and display name
                step        = int_vals_ord[key][2]
                min_v       = int_vals_ord[key][3]
                max_v       = int_vals_ord[key][4]
                default     = int_vals_ord[key][5]
                disp_name   = int_vals_ord[key][6]
                frmt        = int_vals_ord[key][7]

                # Set widget key
                widget_key = f"num_input_{key}_ord"


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

                # Render input 
                values[key] = st.number_input(
                                            disp_name,
                                            key=widget_key,
                                            value=val,
                                            step=step,
                                            min_value=min_v,
                                            max_value=max_v,
                                            on_change=reset_ord_plot,
                                            format = frmt,
                                            )
                

            
    # Generate and display the RUC
    if func_ord is not None:

        # Create columns for organization
        col_ord_out_1, col_ord_out_2, col_ord_out_3, col_ord_out_4, col_ord_out_5, __ = st.columns([1, 1, 1, 1, 1, 3])

        # Create the generate RUC button
        with col_ord_out_1:
            st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
            generate_clicked_ord = st.button("Generate RUC", key='gen_button_ord')
            st.write("")

        # Create the gridline checkbox
        with col_ord_out_2:

            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
            show_grid_ord = st.checkbox("Show Grid Lines", value=True, key='grid_check_ord')
            st.write("")

        # Create fiber color selector
        with col_ord_out_3:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Fiber')

            # Set the fiber color selector
            if 'fiber_color_ord' not in st.session_state:
                st.session_state['fiber_color_ord'] = def_color
            fib_color = st.selectbox(
                                    "Fiber Color",
                                    color_list,
                                    key = 'fiber_color_ord'
                                    )
            st.write("")
            
        # Create matrix color selector
        with col_ord_out_4:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Matrix')

            # Set the matrix color selector
            if 'matrix_color_ord' not in st.session_state:
                st.session_state['matrix_color_ord'] = def_color
            mat_color = st.selectbox(
                                    "Matrix Color",
                                    color_list,
                                    key = 'matrix_color_ord'
                                    )
            st.write("")

        # Create interface color selector
        if interface_ord:
            with col_ord_out_5:

                # Get the color list and default color
                color_list, def_color = UI_Definitions('Interface')

                # Set the matrix color selector
                if 'interface_color_ord' not in st.session_state:
                    st.session_state['interface_color_ord'] = def_color
                mat_color = st.selectbox(
                                        "Interface Color",
                                        color_list,
                                        key = 'interface_color_ord'
                                        )
                st.write("")

        # Create microstructure
        if generate_clicked_ord:

            func_values = {}
            flag = 0
            for key in values.keys():
                if key in def_list_ord[def_opt_ord]['Inputs']:
                    func_values[key] = values[key]

                if interface_ord:
                    if key in int_vals_ord.keys():
                        func_values[key] = values[key]

            st.session_state['mask_ord'] = func_ord(**func_values)

        # Plot
        if 'mask_ord' in st.session_state:

            # Get the mask
            mask, out = st.session_state['mask_ord']

            # Create the plot
            if interface_ord:
                fig = Plot(mask, st.session_state['fiber_color_ord'], st.session_state['matrix_color_ord'], show_grid_ord, st.session_state['interface_color_ord'])
            else:
                fig = Plot(mask, st.session_state['fiber_color_ord'], st.session_state['matrix_color_ord'], show_grid_ord)

            # Create columns for visualalization and data
            col_ord_plot_1, col_ord_plot_2, __ = st.columns([2, 2, 4])

            # Display the microstruture
            with col_ord_plot_1:
                st.plotly_chart(fig, width='content', key = 'plot_ord')

            # Create table with actual microstructure properties
            with col_ord_plot_2:
                if interface_ord:
                    data = {
                            'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Interface Volume Fraction', 'Interface Thickness', 'Subcells in X', 'Subcells in Y'],
                            'Value':[out['VF'], out['R'], out['VI'], out['RI'], out['NB'], out['NG']],
                            }

                else:
                    data = {
                            'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Subcells in X', 'Subcells in Y'],
                            'Value':[out['VF'], out['R'], out['NB'], out['NG']],
                            }
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_ord', hide_index=True) 

            # Create Files
            csv_data = WriteCSV(mask)
            ruc_data = WriteRUC(mask)

            # Create columns for downloading data
            st.write('Download the RUC:')
            __, col_ord_dwnld_1, __ = st.columns([0.05, 2, 5.95])

            # Create download buttons
            with col_ord_dwnld_1:
                # Download to CSV
                st.download_button(
                                label="Download  CSV",
                                data=csv_data,
                                file_name="ruc.csv",
                                mime="text/csv",
                                key="download_csv_ord"
                                )

                # Download for *RUC
                st.download_button(
                                label="Download *RUC File",
                                data=ruc_data,
                                file_name="ruc_data.txt",
                                mime="text/plain",
                                key="download_ruc_ord"
                                )

#------------------------------------------------------------------------------------------------------------------------------------------
#   RANDOM MICROSTRUCTURE
#   Generate a periodic or non-periodic random microstructure.
#------------------------------------------------------------------------------------------------------------------------------------------

# Random Microstructure Generation   
with tab_rand:

    # Create header
    st.markdown("## Random RVE Generator")

    # Create description
    st.markdown("""Generate a random Repeating Unit Cell (RUC) microstructure based on user-defined parameters. 
                Select the input parameters to visualize and download the RUC data.""")
    
    # Initialize Function
    func_rand = None

    # Create columns for microstructure and definition selection
    col_rand_alg_1, col_rand_alg_2 = st.columns([1, 1])

    # Create input for microstructure type
    with col_rand_alg_1:
        # Get the list of available algorithms
        algo_list = UI_Definitions('AlgorithmList')

        # Algorithm selector
        alg_opt_rand = st.selectbox(
            "Select an algorithm:",
            algo_list,
            key='alg_opt_rand'
        )

        # Checkbox lives here
        def reset_rand_plot():
            if 'select_ruc_rand' in st.session_state:
                del st.session_state['select_ruc_rand']
                del st.session_state['masks_rand']
                generate_random = False
        interface_rand = st.checkbox("Include Interface", key='check_interface_rand', on_change=reset_rand_plot)

    with col_rand_alg_2:
        # Only show dropdown if checkbox is active
        if interface_rand:
            interface_opts, int_vals_rand_all = UI_Definitions('InterfaceOptions')

            interface_rand_ord = st.selectbox(
                "Interface Definition:",
                interface_opts,
                key='interface_opt_rand'
            )
        else:
            # create an empty placeholder to keep row alignment clean
            st.write("")   # this creates a blank element but does NOT shift layout
        

    # Soft Body Dynamics
    if alg_opt_rand == "Soft Body Dynamics":
        def_vals_rand, func_rand = UI_Definitions("SBD")
        
    # Separate user inputs
    st.markdown('''---''')

    # Create user inputs
    if func_rand is not None:

        # Get the function and initalize values
        values = {}

        # Separeate numeric inputs into two columns
        col_rand_inp_1, col_rand_inp_2 = st.columns([1, 1])

        # Create numeric inputs
        for key in def_vals_rand.keys():

            # Determine column
            colnum = col_rand_inp_1 if def_vals_rand[key][0] == 1 else col_rand_inp_2

            with colnum:

                # Float and Integer Inputs
                if def_vals_rand[key][1] in ['float', 'int']:

                    # Get min, max, step, default, and display name
                    step        = def_vals_rand[key][2]
                    min_v       = def_vals_rand[key][3]
                    max_v       = def_vals_rand[key][4]
                    default     = def_vals_rand[key][5]
                    disp_name   = def_vals_rand[key][6]
                    frmt        = def_vals_rand[key][7]

                    # Set widget key
                    widget_key = f"num_input_{key}_rand"

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

                    # Render input 
                    values[key] = st.number_input(
                                                disp_name,
                                                key=widget_key,
                                                value=val,
                                                step=step,
                                                min_value=min_v,
                                                max_value=max_v,
                                                format= frmt
                                                )

                # Discrete (Drop Down Menu) Inputs
                elif def_vals_rand[key][1] in ['disc']:
                    # Set widget key
                    widget_key = f"disc_input_{key}_rand"

                    # Get options and name
                    opts = def_vals_rand[key][2]
                    disp_name = def_vals_rand[key][3]

                    # Render input
                    values[key] = st.selectbox(
                                            disp_name,
                                            opts,
                                            key=widget_key,
                                            )
                    
        # Create user inputs
        if interface_rand:

            # Get the correct inputs
            int_vals_rand = int_vals_rand_all[st.session_state['interface_opt_rand']]

            # Create numeric inputs
            for key in int_vals_rand.keys():

                # Determine column
                colnum = col_rand_inp_1 if int_vals_rand[key][0] == 1 else col_rand_inp_2

                with colnum:

                    # Get min, max, step, default, and display name
                    step        = int_vals_rand[key][2]
                    min_v       = int_vals_rand[key][3]
                    max_v       = int_vals_rand[key][4]
                    default     = int_vals_rand[key][5]
                    disp_name   = int_vals_rand[key][6]

                    # Set widget key
                    widget_key = f"num_input_{key}_rand"

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

                    # Render input 
                    values[key] = st.number_input(
                                                disp_name,
                                                key=widget_key,
                                                value=val,
                                                step=step,
                                                min_value=min_v,
                                                max_value=max_v,
                                                )

    # Create columns for organization
    col_rand_gen_1, __, __ = st.columns([1, 1, 9])

    # Create the generate RUC button
    with col_rand_gen_1:
        st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
        generate_random = st.button("Generate RUCs", key = 'gen_button_rand')
        st.write("")

    # Generate RUCs
    if generate_random:
        with st.spinner("Generating RUCs..."):
            masks = func_rand(**values)

        # Save generated RUCs to session state
        st.session_state['masks_rand'] = masks
        st.session_state['select_ruc_rand'] = masks[0][0]  # default selection

    # Only display RUC selection if we have generated RUCs
    if 'masks_rand' in st.session_state:
        masks = st.session_state['masks_rand']

        # Organize into columns
        col_rand_out_1, col_rand_out_2, col_rand_out_3, col_rand_out_4, col_rand_out_5, __ = st.columns([1, 1, 1, 1, 1, 3])

        with col_rand_out_1:

            # Create Select Box to choose RUC
            rve_names = [name for name,_,_ in masks]
            selected_rve = st.selectbox(
                                        "Select an RUC to visualize:", 
                                        rve_names,
                                        key = 'select_ruc_rand',
                                        )
            st.write("")
            
        with col_rand_out_2:

            # Gridline checkbox
            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
            show_grid_rand = st.checkbox("Show Grid Lines", value=True, key='grid_check_rand')
            st.write("")

        # Create fiber color selector
        with col_rand_out_3:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Fiber')

            # Set the fiber color selector
            if 'fiber_color_rand' not in st.session_state:
                st.session_state['fiber_color_rand'] = def_color
            fib_color = st.selectbox(
                                    "Fiber Color",
                                    color_list,
                                    key = 'fiber_color_rand'
                                    )   
            st.write("")
            
        # Create matrix color selector
        with col_rand_out_4:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Matrix')

            # Set the matrix color selector
            if 'matrix_color_rand' not in st.session_state:
                st.session_state['matrix_color_rand'] = def_color
            mat_color = st.selectbox(
                                    "Matrix Color",
                                    color_list,
                                    key = 'matrix_color_rand'
                                    )   
            st.write("")

        # Create interface color selector
        if interface_rand:
            with col_rand_out_5:

                # Get the color list and default color
                color_list, def_color = UI_Definitions('Interface')

                # Set the matrix color selector
                if 'interface_color_rand' not in st.session_state:
                    st.session_state['interface_color_rand'] = def_color
                mat_color = st.selectbox(
                                        "Interface Color",
                                        color_list,
                                        key = 'interface_color_rand'
                                        )
                st.write("")
            
        # Only create plot of an RUC exists
        if 'select_ruc_rand' in st.session_state:

            # Get the selected mask
            for name, mask, out in masks:
                if name == st.session_state['select_ruc_rand']:
                    selected_mask = mask
                    selected_out = out
                    break
            
            # Create the plot
            if interface_rand:
                fig = Plot(selected_mask, st.session_state['fiber_color_rand'], st.session_state['matrix_color_rand'], show_grid_rand, st.session_state['interface_color_rand'])
            else:
                fig = Plot(selected_mask, st.session_state['fiber_color_rand'], st.session_state['matrix_color_rand'], show_grid_rand)

            # Create columns for visualalization and data
            col_rand_plot_1, col_rand_plot_2, __ = st.columns([2, 2, 4])

            # Display the microstruture
            with col_rand_plot_1:
                st.plotly_chart(fig, width='content', key='plot_rand')

            # Create table with actual microstructure properties
            with col_rand_plot_2:
                if interface_rand:
                    data = {
                            'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Interface Volume Fraction', 'Interface Thickness', 'Subcells in X', 'Subcells in Y'],
                            'Value':[out['VF'], out['R'], out['VI'], out['RI'], out['NB'], out['NG']],
                            }

                else:
                    data = {
                            'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Subcells in X', 'Subcells in Y'],
                            'Value':[out['VF'], out['R'], out['NB'], out['NG']],
                            }
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_rand', hide_index=True)

            # Generate summay information
            st.markdown("---")
            st.write("Summary:")

            # Calculate averages and standard deviations
            if interface_rand:
                out_sum = {
                        'Fiber Volume Fraction':[],
                        'Fiber Radius':[],
                        'Interface Volume Fraction':[],
                        'Interface Thickness':[],
                        'Subcells in X':[],
                        'Subcells in Y':[],
                        }
                out_sum_keys = ['VF', 'R', 'VI', 'RI', 'NB', 'NG']

            else:
                out_sum = {
                        'Fiber Volume Fraction':[],
                        'Fiber Radius':[],
                        'Subcells in X':[],
                        'Subcells in Y':[],
                        }
                out_sum_keys = ['VF', 'R', 'NB', 'NG']
            
            for name, mask, out in masks:
                for i, key in enumerate(out_sum.keys()):
                    out_sum[key].append(out[out_sum_keys[i]])

            for key in out_sum.keys():
                out_sum[key] = [np.average(out_sum[key]), np.std(out_sum[key])]

            # Format into dataframe
            rows = []
            for key, value_list in out_sum.items():
                v1, v2 = value_list  # unpack the two items
                rows.append([key, v1, v2])

            df_out = pd.DataFrame(rows, columns=["Parameter", "Average", "Standard Deviation"])

            # Create columns for the summary table
            col_rand_sum_1,  __ = st.columns([4, 4])
            with col_rand_sum_1:
                st.dataframe(df_out, hide_index=True)
        
            # Create columns for downloading data
            st.write('Download the RUCs:')
            __, col_rand_dwnld_1,  __ = st.columns([0.05, 2, 5.95])
                
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
            with col_rand_dwnld_1:
                    st.download_button(
                                    label="Download All to CSV",
                                    data=zip_bytes_csv,
                                    file_name="RVEs.zip",
                                    mime="application/zip",
                                    key = "download_csv_rand"
                                    )

            # Generate RUC Files
            def generate_zip_with_masks_ruc(masks):
                zip_buffer = io.BytesIO()
                all_texts = []

                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                    for name, mask, _ in masks:
                        txt_content = WriteRUC(mask)  # returns string
                        zip_file.writestr(f"{name}.txt", txt_content)

                        # accumulate for the combined file (add a header so it's readable)
                        all_texts.append(f"#--- {name}.txt ---\n{txt_content}\n")

                    # Create combined file
                    combined_text = "\n".join(all_texts)
                    zip_file.writestr("RVE_ALL.txt", combined_text)

                zip_buffer.seek(0)
                return zip_buffer

            zip_bytes_ruc = generate_zip_with_masks_ruc(st.session_state['masks_rand'])

            # Download for *RUC
            with col_rand_dwnld_1:
                st.download_button(
                                label="Download All *RUC Files",
                                data=zip_bytes_ruc,
                                file_name="RVEs_MAC.zip",
                                mime="application/zip",
                                key = "download_ruc_rand"
                                )

            # Download images
            def parse_color_to_rgb(color_str):
                """Safely parse hex, rgb(), rgba(), or named CSS colors."""
                return ImageColor.getrgb(color_str)
            
            def generate_zip_with_all_mask_images(masks, fiber_color, matrix_color, scale=255):
                fiber_rgb = parse_color_to_rgb(fiber_color)
                matrix_rgb = parse_color_to_rgb(matrix_color)

                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipf:

                    for name, mask, _ in masks:

                        H, W = mask.shape
                        img = np.zeros((H, W, 3), dtype=np.uint8)

                        img[mask == 1] = fiber_rgb
                        img[mask == 2] = matrix_rgb

                        img_pil = Image.fromarray(img)
                        img_bytes = io.BytesIO()
                        img_pil.save(img_bytes, format="PNG")
                        img_bytes.seek(0)

                        zipf.writestr(f"{name}.png", img_bytes.read())

                zip_buffer.seek(0)
                return zip_buffer
            
            # Download for *RUC
            with col_rand_dwnld_1:
                zip_bytes = generate_zip_with_all_mask_images(
                                                            masks,
                                                            st.session_state['fiber_color_rand'],
                                                            st.session_state['matrix_color_rand']
                                                            )
                
                st.download_button(
                                label="Download All Images",
                                data=zip_bytes,
                                file_name="All_RVE_Images.zip",
                                mime="application/zip",
                                )   

#------------------------------------------------------------------------------------------------------------------------------------------
#   GENERATED MICROSTRUCTURE FROM IMAGE
#   Generate a microstructure voxelation from a segmented image.
#------------------------------------------------------------------------------------------------------------------------------------------

# Segmented Image Microstructure Generation
with tab_img:

    # Create Header
    st.markdown("## Generate RUC from Segmented Image")

    # Create description
    st.markdown("""Generate a RUC microstructure from a segmented image. 
                Upload a segmented image and specify parameters to visualize and download the RUC data.
                Create a statistically equivalent RUC using the optimization tool.""")

    # Allow file upload
    uploaded_file = st.file_uploader("Choose a file", type=["png","jpg"], key = 'file_upload_img')

    # Display the image
    if uploaded_file is not None:
        
        #Read imate from file 
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR

        # Create columns for organization
        col_img_disp_1, col_img_disp_2, col_img_disp_3 = st.columns([2, 1, 1])

        with col_img_disp_1:

            # Create the image
            st.image(img, channels="BGR")

            # Set color options
            blue_mask  = np.all(img == [255, 0, 0], axis=2)
            green_mask = np.all(img == [0, 255, 0], axis=2)
            red_mask   = np.all(img == [0, 0, 255], axis=2)

            # Find segmented colors
            colors = []
            if np.any(blue_mask):
                colors.append('Blue')
            if np.any(green_mask):
                colors.append('Green')
            if np.any(red_mask):
                colors.append('Red')

        # Reset W and H if reduction size defined
        def on_red_size_change():
            st.session_state.w_img = None
            st.session_state.h_img = None

        # Reset/set reduction size and H if W defined
        def on_w_img_change():
            st.session_state.red_size_img = None
            # If h_img is None or "other", set default of 100
            if st.session_state.h_img is None:
                st.session_state.h_img = 100

        # Reset/set reduction size and W if H defined
        def on_h_img_change():
            st.session_state.red_size_img = None
            # If w_img is None or "other", set default of 100
            if st.session_state.w_img is None:
                st.session_state.w_img = 100

        # Initialize session state variables
        if 'w_img' not in st.session_state:
            st.session_state.w_img = None
        if 'h_img' not in st.session_state:
            st.session_state.h_img = None
        if 'red_size_img' not in st.session_state:
            st.session_state.red_size_img = 0.2

        with col_img_disp_2:

            # Create select box for fiber color
            color = st.selectbox(
                                "Select Fiber Color in Image",
                                colors,
                                key='color_select_img'
                                )

            # Create input for number of subcells in W
            w_img = st.number_input(
                                'RUC Width (subcells)',
                                key='w_img',
                                step=1,
                                min_value=1,
                                max_value=None,
                                on_change=on_w_img_change
                                )

            # Create input for max nub length
            nub_img = st.number_input(
                                    'Max Nub Length (pixels)',
                                    key='nub_img',
                                    step=1,
                                    min_value=1,
                                    max_value=5,
                                    value = 1,
                                    )

            # Create checkbox to remove touching fiber sbucells
            touch_img = st.checkbox("Remove Touching Fiber Subcells", value=True, key='touch_check_img')

        with col_img_disp_3:

            # Create input for reduction size
            red_size = st.number_input(
                                    'Reduction Size',
                                    key='red_size_img',
                                    step=0.01,
                                    min_value=0.01,
                                    max_value=1.,
                                    on_change=on_red_size_change
                                    )

            # Create input for number of subcells in H
            h_img = st.number_input(
                                    'RUC Height (subcells)',
                                    key='h_img',
                                    step=1,
                                    min_value=1,
                                    max_value=None,
                                    on_change=on_h_img_change
                                    )

            # Create input for maximum corner length
            corner_img = st.number_input(
                                        'Max Corner Length (pixels)',
                                        key='corner_img',
                                        step=1,
                                        min_value=1,
                                        max_value=5,
                                        value = 1,
                                        )

        # Create columns for organization
        col_img_gen_1, col_img_gen_2, col_img_gen_3, col_img_gen_4, __ = st.columns([1, 1, 1, 1, 4])

        # Create the generate RUC button
        with col_img_gen_1:
            st.markdown(f'<div style="height:{26}px"></div>', unsafe_allow_html=True)
            generate_clicked_img = st.button("Generate RUC", key = 'gen_button_img')
            st.write("")

        # Create the gridline checkbox
        with col_img_gen_2:
            st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
            show_grid_img = st.checkbox("Show Grid Lines", value=True, key='grid_check_img')
            st.write("")

        # Create fiber color selector
        with col_img_gen_3:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Fiber')

            # Set the fiber color selector
            if 'fiber_color_img' not in st.session_state:
                st.session_state['fiber_color_img'] = def_color
            fib_color = st.selectbox(
                                    "Fiber Color",
                                    color_list,
                                    key = 'fiber_color_img'
                                    )
            st.write("")
            
        # Create matrix color selector
        with col_img_gen_4:

            # Get the color list and default color
            color_list, def_color = UI_Definitions('Matrix')

            # Set the matrix color selector
            if 'matrix_color_img' not in st.session_state:
                st.session_state['matrix_color_img'] = def_color
            mat_color = st.selectbox(
                                    "Matrix Color",
                                    color_list,
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

            # Create input array for fiber idealization
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
            
            # Run the idealization algorithm
            with st.spinner('Generating Microstructure...'):
                st.session_state['mask_img'] = Segmented(Input)

        # Only plot if we have a mask
        if 'mask_img' in st.session_state:

            # Get the mask
            mask, out = st.session_state['mask_img']

            # Create the plot
            fig = Plot(mask, st.session_state['fiber_color_img'], st.session_state['matrix_color_img'], show_grid_img)

            # Create columns for visualalization and data
            col_img_plot_1, col_img_plot_2, __ = st.columns([2, 2, 4])

            # Display the microstruture
            with col_img_plot_1:
                st.plotly_chart(fig, width='content', key = 'plot_img')

            # Create table with actual microstructure properties
            with col_img_plot_2:
                data = {
                        'Property':['VF', 'R', 'NB', 'NG'],
                        'Value':[out['VF'], out['R'], out['NB'], out['NG']],
                        }
                df = pd.DataFrame(data)
                df = df.reset_index(drop=True)
                st.markdown("")
                st.dataframe(df, key = 'out_table_img', hide_index=True) 

            # Create Files
            csv_data = WriteCSV(mask)
            ruc_data = WriteRUC(mask)

            # Create columns for downloading data
            st.write ('Download the RUC:')
            __, col_img_dwnld_1, __ = st.columns([0.05, 2, 5.95])

            with col_img_dwnld_1:
                # Download to CSV
                st.download_button(
                                label="Download  CSV",
                                data=csv_data,
                                file_name="ruc.csv",
                                mime="text/csv",
                                key="download_csv_img"
                                )

                # Download for *RUC
                st.download_button(
                                label="Download *RUC File",
                                data=ruc_data,
                                file_name="ruc_data.txt",
                                mime="text/plain",
                                key="download_ruc_img"
                                )
                
    # Initialize session state to show optimization inputs
    if "show_opt_inputs" not in st.session_state:
        st.session_state.show_opt_inputs = False
        st.session_state.show_opt_img = False

    # Only create optimization inputs if a mask exists
    try:
        # Read the mask
        if st.session_state['mask_img']:

            # Separator for organization
            st.markdown("---")

            # Create the RVE
            create_rve_img = st.button('Create RVE', key = 'create_rve_img')

            if create_rve_img:

                # Set session state
                st.session_state.show_opt_inputs = True
                
            # Create optimization inputs
            if st.session_state.show_opt_inputs:

                # Create columns for microstructure and definition selection
                col_img_alg_1, __= st.columns([1, 1])

                # Create input for microstructure type
                with col_img_alg_1:

                    # Get the list of available algorithms
                    algo_list = UI_Definitions('AlgorithmList')

                    # Create the input
                    alg_opt_img = st.selectbox(
                                            "Select an algorithm:",
                                            algo_list,
                                            key = 'alg_opt_img',
                                            )

                # Soft Body Dynamics
                if alg_opt_img == "Soft Body Dynamics":
                    input_space_img, constants_img, opt_settings_img = UI_Definitions('SBD_Opt')
                    st.session_state['input_space_img'] = input_space_img

                # Create inputs columns
                st.markdown('### Input Space')
                col_img_optin_1, col_img_optin_2, col_img_optin_3, __ = st.columns([1.5, 2, 2, 5.5])

                # Add labels
                with col_img_optin_1:
                    st.write('Parameters')
                with col_img_optin_2:
                    st.write('Lower Bound')
                with col_img_optin_3:
                    st.write('Upper Bound')
                
                # Create inputs
                for i, key in enumerate(input_space_img.keys()):
                    enabled_key = f"optin_check_{i}_img"
                    low_key     = f"optin_num_{i}_low_img"
                    high_key    = f"optin_num_{i}_high_img"

                    step    = input_space_img[key][1]
                    min_v   = input_space_img[key][2]
                    max_v   = input_space_img[key][3]
                    low_def = input_space_img[key][4]
                    high_def= input_space_img[key][5]
                    disp    = input_space_img[key][6]
                    frmt    = input_space_img[key][7]

                    # Create row
                    with st.container():
                        col1, col2, col3, __ = st.columns([1.5, 2, 2, 5.5])

                        # Checkbox (enable/disable row)
                        with col1:
                            enabled = st.checkbox(disp, value=True, key=enabled_key)

                        # Low bound
                        with col2:

                            # Get previous or default
                            low_val = st.session_state.get(low_key, low_def)

                            # Constrain lower bound so it can’t be above the current upper bound
                            current_high = st.session_state.get(high_key, high_def)
                            max_low = current_high

                            # Create input
                            low_val = st.number_input(
                                                    "Low",
                                                    key=low_key,
                                                    value=low_val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_low,
                                                    format=frmt,
                                                    label_visibility="collapsed",
                                                    disabled=not enabled
                                                    )

                        # High bound
                        with col3:

                            # Get previous or default
                            high_val = st.session_state.get(high_key, high_def)

                            # Constrain high bound so it can't be below current low
                            current_low = st.session_state.get(low_key, low_def)
                            min_high = current_low

                            # Create input
                            high_val = st.number_input(
                                                    "High",
                                                    key=high_key,
                                                    value=high_val,
                                                    step=step,
                                                    min_value=min_high,
                                                    max_value=max_v,
                                                    format=frmt,
                                                    label_visibility="collapsed",
                                                    disabled=not enabled
                                                    )

                    # Store the range only if enabled
                    if enabled:
                        values[key] = (low_val, high_val)
                    else:
                        values[key] = None   # or whatever you prefer

                # Create RVE constant columns
                st.markdown('### RVE Definition')
                col_img_rve_1, col_img_rve_2,  __ = st.columns([2, 2, 4])

                # Create numeric inputs
                for key in constants_img.keys():

                    # Determine column
                    colnum = col_img_rve_1 if constants_img[key][0] == 1 else col_img_rve_2

                    with colnum:

                        # Get min, max, step, default, and display name
                        step        = constants_img[key][2]
                        min_v       = constants_img[key][3]
                        max_v       = constants_img[key][4]
                        default     = constants_img[key][5]
                        disp_name   = constants_img[key][6]

                        # Set widget key
                        widget_key = f"opt_rve_{key}_img"

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

                        # Render input 
                        values[key] = st.number_input(
                                                    disp_name,
                                                    key=widget_key,
                                                    value=val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_v,
                                                    )

                # Create optimization settings columns
                st.markdown('### Optimization Settings')
                col_img_opt_1, col_img_opt_2,  __ = st.columns([2, 2, 4])

                # Create numeric inputs
                for key in opt_settings_img.keys():

                    # Determine column
                    colnum = col_img_opt_1 if opt_settings_img[key][0] == 1 else col_img_opt_2

                    with colnum:

                        # Get min, max, step, default, and display name
                        step        = opt_settings_img[key][2]
                        min_v       = opt_settings_img[key][3]
                        max_v       = opt_settings_img[key][4]
                        default     = opt_settings_img[key][5]
                        disp_name   = opt_settings_img[key][6]

                        # Set widget key
                        widget_key = f"opt_opt_{key}_img"

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

                        # Render input 
                        values[key] = st.number_input(
                                                    disp_name,
                                                    key=widget_key,
                                                    value=val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_v,
                                                    )
                        
                # Create button to optimize the RVE
                opt_rve_img = st.button('Optimize RVE', key = 'opt_rve_img')

                if opt_rve_img:
                    
                    # Gather inputs
                    optimization_inputs = {}

                    if alg_opt_img == "Soft Body Dynamics":
                        func_opt_img, func_inp_img = UI_Definitions("SBD_Opt_Run", st.session_state, 'img')

                    # Call the function
                    with st.spinner('Running Optimization...'):

                        # Callback function to display output
                        def ui_callback_img(msg):
                            # Each new message will appear below previous ones
                            # append messages to the scrollable box
                            if 'messages_img' not in st.session_state:
                                st.session_state['messages_img'] = []
                            st.session_state['messages_img'].append(msg)
                            
                            # render all messages inside scrollable div
                            msgs_html = "<br>".join(st.session_state['messages_img'])
                            progress_box.markdown(f'<div class="scrollable-box">{msgs_html}</div>', unsafe_allow_html=True)

                        # Create scrollable style
                        scrollable_style = """
                            <style>
                            .scrollable-box {
                                max-height: 600px;
                                overflow-y: auto;
                                border: 1px solid #ddd;
                                padding: 5px;
                                background-color: #f9f9f9;
                            }
                            </style>
                        """

                        # Display progress bar
                        st.markdown(scrollable_style, unsafe_allow_html=True)
                        st.session_state['messages_img'] = []
                        progress_box = st.empty()

                        # Run the function
                        best_mask, best_out, best_error = func_opt_img(**func_inp_img, callback = ui_callback_img)

                        # Get best values
                        best_out['Error'] = best_error
                        st.session_state['mask_opt_img'] = [best_mask, best_out]

                        # Reset progress bar
                        progress_box.empty()

                        # Set sessions state variable
                        st.session_state.show_opt_img = True

                if st.session_state.show_opt_img:

                    # Create columns for organization
                    col_opt_out_1, col_opt_out_2, col_opt_out_3, __ = st.columns([1, 1, 1, 3])

                    # Create the gridline checkbox
                    with col_opt_out_1:
                        st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
                        show_grid_opt_img = st.checkbox("Show Grid Lines", value=True, key='grid_check_opt_img')
                        st.write("")

                    # Create fiber color selector
                    with col_opt_out_2:

                        # Get the color list and default color
                        color_list, def_color = UI_Definitions('Fiber')

                        # Set the fiber color selector
                        if 'fiber_color_opt_img' not in st.session_state:
                            st.session_state['fiber_color_opt_img'] = def_color
                        fib_color = st.selectbox(
                                                "Fiber Color",
                                                color_list,
                                                key = 'fiber_color_opt_img'
                                                )
                        st.write("")
                        
                    # Create matrix color selector
                    with col_opt_out_3:

                        # Get the color list and default color
                        color_list, def_color = UI_Definitions('Matrix')

                        # Set the matrix color selector
                        if 'matrix_color_opt_img' not in st.session_state:
                            st.session_state['matrix_color_opt_img'] = def_color
                        mat_color = st.selectbox(
                                                "Matrix Color",
                                                color_list,
                                                key = 'matrix_color_opt_img'
                                                )
                        st.write("")

                    # Plot
                    if 'mask_opt_img' in st.session_state:

                        # Get the mask
                        mask, out = st.session_state['mask_opt_img']

                        # Create the plot
                        fig = Plot(mask, st.session_state['fiber_color_opt_img'], st.session_state['matrix_color_opt_img'], show_grid_opt_img)

                        # Create columns for visualalization and data
                        col_opt_plot_1, col_opt_plot_2, __ = st.columns([2, 2, 4])

                        # Display the microstruture
                        with col_opt_plot_1:
                            st.plotly_chart(fig, width='content', key = 'plot_opt')

                        # Create table with actual microstructure properties
                        with col_opt_plot_2:
                            data = {
                                    'Property':['VF', 'R', 'NB', 'NG', 'Error'],
                                    'Value':[out['VF'], out['R'], out['NB'], out['NG'], out['Error']],
                                    }
                            df = pd.DataFrame(data)
                            df = df.reset_index(drop=True)
                            st.markdown("")
                            st.dataframe(df, key = 'out_table_opt', hide_index=True) 

                        # Create Files
                        csv_data = WriteCSV(mask)
                        ruc_data = WriteRUC(mask)

                        # Create columns for downloading data
                        st.write('Download the RUC:')
                        __, col_opt_img_dwnld_1, __ = st.columns([0.05, 2, 5.95])

                        with col_opt_img_dwnld_1:
                            # Download to CSV
                            st.download_button(
                                            label="Download  CSV",
                                            data=csv_data,
                                            file_name="ruc.csv",
                                            mime="text/csv",
                                            key="download_csv_opt_img"
                                            )

                            # Download for *RUC
                            st.download_button(
                                            label="Download *RUC File",
                                            data=ruc_data,
                                            file_name="ruc_data.txt",
                                            mime="text/plain",
                                            key="download_ruc_opt_img"
                                            )
    except:
        pass

#------------------------------------------------------------------------------------------------------------------------------------------
#   VISUALIZE AN RUC
#   Visualize and RUC from a *.csv or *.mac file.
#------------------------------------------------------------------------------------------------------------------------------------------
#                 
# RUC Visualizer
with tab_viz:
    # Create header
    st.markdown("## RUC Visualizer")

    # Create description
    st.markdown("Upload a CSV or *RUC file to visualize the microstructure. Create a statistically equivalent RUC using the optimization tool.")

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

            # Create columns for organization
            col_viz_disp_1, col_viz_disp_2, col_viz_disp_3, col_viz_disp_4, __ = st.columns([1, 1, 1, 1, 2])

            # Create the gridline checkbox
            with col_viz_disp_1:
                st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
                show_grid_viz = st.checkbox("Show Grid Lines", value=True, key='grid_check_viz')
                st.write("")

            # Create fiber selector
            with col_viz_disp_2:

                # Get the color list and default color
                color_list, def_color = UI_Definitions('Fiber')

                # Set the fiber color selector
                if 'fiber_color_viz' not in st.session_state:
                    st.session_state['fiber_color_viz'] = def_color
                fib_color = st.selectbox(
                                        "Fiber Color",
                                        color_list,
                                        key = 'fiber_color_viz'
                                        )
                st.write("")
                
            # Create matrix color selector
            with col_viz_disp_3:

                # Get the color list and default color
                color_list, def_color = UI_Definitions('Matrix')

                # Set the matrix color selector
                if 'matrix_color_viz' not in st.session_state:
                    st.session_state['matrix_color_viz'] = def_color
                mat_color = st.selectbox(
                                        "Matrix Color",
                                        color_list,
                                        key = 'matrix_color_viz'
                                        )
                st.write("")

            interface_viz = False
            if 'VI' in out.keys():
                interface_viz = True
                with col_viz_disp_4:

                    # Get the color list and default color
                    color_list, def_color = UI_Definitions('Interface')

                    # Set the matrix color selector
                    if 'interface_color_viz' not in st.session_state:
                        st.session_state['interface_color_viz'] = def_color
                    int_color = st.selectbox(
                                            "Interface Color",
                                            color_list,
                                            key = 'interface_color_viz'
                                            )
                    st.write("")


            # Only plot if we have a mask
            if 'mask_viz' in st.session_state:

                # Get the mask
                mask = st.session_state['mask_viz']

                # Create the plot
                if interface_viz:
                    fig = Plot(mask, st.session_state['fiber_color_viz'], st.session_state['matrix_color_viz'], show_grid_viz, st.session_state['interface_color_viz'])
                else:
                    fig = Plot(mask, st.session_state['fiber_color_viz'], st.session_state['matrix_color_viz'], show_grid_viz)

                # Create columns for visualalization and data
                col_viz_plot_1, col_viz_plot_2, __ = st.columns([1, 1, 2])

                # Display the microstruture
                with col_viz_plot_1:
                    st.plotly_chart(fig, width='content', key='plot_viz')

                # Create table with actual microstructure properties
                with col_viz_plot_2:
                    if interface_rand:
                        data = {
                                'Property':['Fiber Volume Fraction', 'Interface Volume Fraction', 'Subcells in X', 'Subcells in Y'],
                                'Value':[out['VF'],  out['VI'],  out['NB'], out['NG']],
                                }

                    else:
                        data = {
                                'Property':['Fiber Volume Fraction',  'Subcells in X', 'Subcells in Y'],
                                'Value':[out['VF'], out['NB'], out['NG']],
                                }
                    df = pd.DataFrame(data)
                    df = df.reset_index(drop=True)
                    st.markdown("")
                    st.dataframe(df, key = 'out_table_viz', hide_index=True) 

                # Create Files
                csv_data = WriteCSV(mask)
                ruc_data = WriteRUC(mask)

                # Create columns for downloading data
                st.write("Download the RUC:")
                __, col_viz_dwnld_1,  __ = st.columns([0.05, 2, 5.95])

                with col_viz_dwnld_1:
                    # Download to CSV
                    st.download_button(
                                    label="Download CSV",
                                    data=csv_data,
                                    file_name="ruc.csv",
                                    mime="text/csv",
                                    key = "download_csv_viz"
                                    )

                    # Download for *RUC
                    st.download_button(
                                    label="Download *RUC File",
                                    data=ruc_data,
                                    file_name="ruc_data.txt",
                                    mime="text/plain",
                                    key="download_ruc_viz"
                                    )

    # Initialize session state to show optimization inputs
    if "show_viz_inputs" not in st.session_state:
        st.session_state.show_viz_inputs = False
        st.session_state.show_opt_viz = False

    # Only create optimization inputs if a mask exists
    try:
        if 'mask_viz' in st.session_state:

            # Get the mask
            mask = st.session_state['mask_viz']

            # Determine if characterization is possible
            try:
                RandomCharacterization(mask, nbins = 10)
            except:
                st.session_state.show_viz_inputs = False
                st.stop()

            # Separator for organization
            st.markdown("---")

            # Create the RVE Button
            create_rve_viz = st.button('Create RVE', key = 'create_rve_viz')

            if create_rve_viz:

                # Set session state
                st.session_state.show_viz_inputs = True

            if st.session_state.show_viz_inputs:

                # Create columns for microstructure and definition selection
                col_viz_alg_1, __= st.columns([1, 1])

                # Create input for microstructure type
                with col_viz_alg_1:

                    # Get the list of available algorithms
                    algo_list = UI_Definitions('AlgorithmList')

                    alg_opt_viz = st.selectbox(
                                            "Select an algorithm:",
                                            algo_list,
                                            key = 'alg_opt_viz',
                                            )

                # Soft Body Dynamics
                if alg_opt_viz == "Soft Body Dynamics":
                    input_space_viz, constants_viz, opt_settings_viz = UI_Definitions('SBD_Opt')
                    st.session_state['input_space_viz'] = input_space_viz
                    
                # Create inputs columns
                st.markdown('### Input Space')
                col_viz_optin_1, col_viz_optin_2, col_viz_optin_3, __ = st.columns([1.5, 2, 2, 5.5])

                # Add lables
                with col_viz_optin_1:
                    st.write('Parameters')
                with col_viz_optin_2:
                    st.write('Lower Bound')
                with col_viz_optin_3:
                    st.write('Upper Bound')
                
                for i, key in enumerate(input_space_viz.keys()):
                    enabled_key = f"optin_check_{i}_viz"
                    low_key     = f"optin_num_{i}_low_viz"
                    high_key    = f"optin_num_{i}_high_viz"

                    step    = input_space_viz[key][1]
                    min_v   = input_space_viz[key][2]
                    max_v   = input_space_viz[key][3]
                    low_def = input_space_viz[key][4]
                    high_def= input_space_viz[key][5]
                    disp    = input_space_viz[key][6]
                    frmt    = input_space_viz[key][7]

                    # Create row
                    with st.container():
                        col1, col2, col3, __ = st.columns([1.5, 2, 2, 5.5])

                        # Checkbox (enable/disable row)
                        with col1:
                            enabled = st.checkbox(disp, value=True, key=enabled_key)

                        # Low bound
                        with col2:
                            # Get previous or default
                            low_val = st.session_state.get(low_key, low_def)

                            # Constrain lower bound so it can’t be above the current upper bound
                            current_high = st.session_state.get(high_key, high_def)
                            max_low = current_high

                            # Create input
                            low_val = st.number_input(
                                                    "Low",
                                                    key=low_key,
                                                    value=low_val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_low,
                                                    format=frmt,
                                                    label_visibility="collapsed",
                                                    disabled=not enabled
                                                    )   

                        # High bound
                        with col3:

                            # Get previous or default
                            high_val = st.session_state.get(high_key, high_def)

                            # Constrain high bound so it can't be below current low
                            current_low = st.session_state.get(low_key, low_def)
                            min_high = current_low

                            # Create input
                            high_val = st.number_input(
                                                    "High",
                                                    key=high_key,
                                                    value=high_val,
                                                    step=step,
                                                    min_value=min_high,
                                                    max_value=max_v,
                                                    format=frmt,
                                                    label_visibility="collapsed",
                                                    disabled=not enabled
                                                    )

                    # Store the range only if enabled
                    if enabled:
                        values[key] = (low_val, high_val)
                    else:
                        values[key] = None   # or whatever you prefer

                # Create RVE constant columns
                st.markdown('### RVE Definition')
                col_viz_rve_1, col_viz_rve_2,  __ = st.columns([2, 2, 4])

                # Create numeric inputs
                for key in constants_viz.keys():

                    # Determine column
                    colnum = col_viz_rve_1 if constants_viz[key][0] == 1 else col_viz_rve_2

                    with colnum:

                        # Get min, max, step, default, and display name
                        step        = constants_viz[key][2]
                        min_v       = constants_viz[key][3]
                        max_v       = constants_viz[key][4]
                        default     = constants_viz[key][5]
                        disp_name   = constants_viz[key][6]

                        # Set widget key
                        widget_key = f"opt_rve_{key}_viz"

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

                        # Render input 
                        values[key] = st.number_input(
                                                    disp_name,
                                                    key=widget_key,
                                                    value=val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_v,
                                                    )

                # Create optimization settings columns
                st.markdown('### Optimization Settings')
                col_viz_opt_1, col_viz_opt_2,  __ = st.columns([2, 2, 4])

                # Create numeric inputs
                for key in opt_settings_viz.keys():

                    # Determine column
                    colnum = col_viz_opt_1 if opt_settings_viz[key][0] == 1 else col_viz_opt_2

                    with colnum:

                        # Get min, max, step, default, and display name
                        step        = opt_settings_viz[key][2]
                        min_v       = opt_settings_viz[key][3]
                        max_v       = opt_settings_viz[key][4]
                        default     = opt_settings_viz[key][5]
                        disp_name   = opt_settings_viz[key][6]

                        # Set widget key
                        widget_key = f"opt_opt_{key}_viz"

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

                        # Render input 
                        values[key] = st.number_input(
                                                    disp_name,
                                                    key=widget_key,
                                                    value=val,
                                                    step=step,
                                                    min_value=min_v,
                                                    max_value=max_v,
                                                    )

                # Create button to optimize the RVE
                opt_rve_viz = st.button('Optimize RVE', key = 'opt_rve_viz')

                if opt_rve_viz:
                    # Gather inputs
                    optimization_inputs = {}

                    if alg_opt_viz == "Soft Body Dynamics":
                        func_opt_viz, func_inp_viz = UI_Definitions("SBD_Opt_Run", st.session_state, 'viz')
                        
                    # Call the function
                    with st.spinner('Running Optimization...'):

                        # Callback function to display output
                        def ui_callback_viz(msg):
                            # Each new message will appear below previous ones
                            # append messages to the scrollable box
                            if 'messages_viz' not in st.session_state:
                                st.session_state['messages_viz'] = []
                            st.session_state['messages_viz'].append(msg)
                            
                            # render all messages inside scrollable div
                            msgs_html = "<br>".join(st.session_state['messages_viz'])
                            progress_box.markdown(f'<div class="scrollable-box">{msgs_html}</div>', unsafe_allow_html=True)

                        # Create scrollable style
                        scrollable_style = """
                            <style>
                            .scrollable-box {
                                max-height: 600px;
                                overflow-y: auto;
                                border: 1px solid #ddd;
                                padding: 5px;
                                background-color: #f9f9f9;
                            }
                            </style>
                        """

                        # Display progress bar
                        st.markdown(scrollable_style, unsafe_allow_html=True)
                        st.session_state['messages_viz'] = []
                        progress_box = st.empty()

                        # Run the function
                        best_mask, best_out, best_error = func_opt_viz(**func_inp_viz, callback = ui_callback_viz)
                        
                        # Get best values
                        best_out['Error'] = best_error
                        st.session_state['mask_opt_viz'] = [best_mask, best_out]
                        

                        # Reset progress bar
                        progress_box.empty()

                        # Set session state
                        st.session_state.show_opt_viz = True

                if st.session_state.show_opt_viz:

                    # Create columns for organization
                    col_opt_viz_out_1, col_opt_viz_out_2, col_opt_viz_out_3, col_opt_viz_out_4, __ = st.columns([1, 1, 1, 1, 2])

                    # Create the gridline checkbox
                    with col_opt_viz_out_1:
                        st.markdown(f'<div style="height:{36}px"></div>', unsafe_allow_html=True)
                        show_grid_opt_viz = st.checkbox("Show Grid Lines", value=True, key='grid_check_opt_viz')
                        st.write("")

                    # -Create fiber color selector
                    with col_opt_viz_out_2:

                        # Get the color list and default color
                        color_list, def_color = UI_Definitions('Fiber')

                        # Set the fiber color selector
                        if 'fiber_color_opt_viz' not in st.session_state:
                            st.session_state['fiber_color_opt_viz'] = def_color
                        fib_color = st.selectbox(
                                                "Fiber Color",
                                                color_list,
                                                key = 'fiber_color_opt_viz'
                                                )
                        st.write("")    
                        
                    # Create matrix color selector
                    with col_opt_viz_out_3:

                        # Get the color list and default color
                        color_list, def_color = UI_Definitions('Matrix')

                        # Set the matrix color selector
                        if 'matrix_color_opt_viz' not in st.session_state:
                            st.session_state['matrix_color_opt_viz'] = def_color
                        mat_color = st.selectbox(
                                                "Matrix Color",
                                                color_list,
                                                key = 'matrix_color_opt_viz'
                                                )
                        st.write("")

                    

                    # Plot
                    if 'mask_opt_viz' in st.session_state:

                        # Get the mask
                        mask, out = st.session_state['mask_opt_viz']

                        if np.max(mask) == 3:
                            # Create interface color selector
                            with col_opt_viz_out_4:

                                # Get the color list and default color
                                color_list, def_color = UI_Definitions('Interface')

                                # Set the matrix color selector
                                if 'interface_color_opt_viz' not in st.session_state:
                                    st.session_state['interface_color_opt_viz'] = def_color
                                mat_color = st.selectbox(
                                                        "Matrix Color",
                                                        color_list,
                                                        key = 'interface_color_opt_viz'
                                                        )
                                st.write("")

                        # Create the plot
                        if np.max(mask) == 2:
                            fig = Plot(mask, st.session_state['fiber_color_opt_viz'], st.session_state['matrix_color_opt_viz'], show_grid_opt_viz)
                        elif np.max(mask) == 3:
                            fig = Plot(mask, st.session_state['fiber_color_opt_viz'], st.session_state['matrix_color_opt_viz'], show_grid_opt_viz, st.session_state['interface_color_opt_viz'])

                        # Create columns for visualalization and data
                        col_opt_plot_1, col_opt_plot_2, __ = st.columns([2, 2, 4])

                        # Display the microstruture
                        with col_opt_plot_1:
                            st.plotly_chart(fig, width='content', key = 'plot_opt_viz')

                        # Create table with actual microstructure properties
                        with col_opt_plot_2:
                            if np.max(mask) == 2:
                                data = {
                                        'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Subcells in X', 'Subcells in Y', 'Error'],
                                        'Value':[out['VF'], out['R'], out['NB'], out['NG'], out['Error']]
                                        }
                            elif np.max(mask) == 3:
                                data = {
                                        'Property':['Fiber Volume Fraction', 'Fiber Radius', 'Interface Volume Fraction', 'Interface Thickness', 'Subcells in X', 'Subcells in Y', 'Error'],
                                        'Value':[out['VF'], out['R'], out['VI'], out['I'], out['NB'], out['NG'], out['Error']]
                                        }
                            df = pd.DataFrame(data)
                            df = df.reset_index(drop=True)
                            st.markdown("")
                            st.dataframe(df, key = 'out_table_opt_viz', hide_index=True) 

                        # Create Files
                        csv_data = WriteCSV(mask)
                        ruc_data = WriteRUC(mask)

                        # Create columns for downloading data
                        st.write('Download the RUC:')
                        __, col_opt_viz_dwnld_1, __ = st.columns([0.05, 2, 5.95])

                        
                        with col_opt_viz_dwnld_1:
                            # Download to CSV
                            st.download_button(
                                            label="Download  CSV",
                                            data=csv_data,
                                            file_name="ruc.csv",
                                            mime="text/csv",
                                            key="download_csv_opt_viz"
                                            )

                            # Download for *RUC
                            st.download_button(
                                            label="Download *RUC File",
                                            data=ruc_data,
                                            file_name="ruc_data.txt",
                                            mime="text/plain",
                                            key="download_ruc_opt_viz"
                                            )

    except:
        pass