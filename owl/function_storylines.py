import numpy as np
import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle
from fnmatch import fnmatch
from matplotlib.patches import Patch
from collections import Counter
from scipy.stats import chi2
from matplotlib.patches import Ellipse, Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def board_benchmark_drivers(board_df,selrate_df,acronyms_drivers,tolerance,thre_ERA5,thre_CMIP6,region,months_code,warming,varspecs,plot_path):
    
    """
        Produce board of the selected drivers for ERA5 and CMIP6
        
        Parameters:
        
            board_df: pd dataframe
                scores as a result of feature selection 
            acronyms_drivers: pd dataframe
                selected drivers with acronyms
            tolerance: float
                range of acceptance below the threshold to be accepted anyway
            thre_ERA5: float
                threshold for a drivers to be selected in ERA5
            thre_CMIP6: float
                same as above, but for CMIp6
            region: str
                region considered (for the title of the plot)
            months_code: str
                codes of the months (initials)
            warming: str
                which warming is this plot related to?
            plot_path: str
                path where you want to save the plot
        Returns:
            plot in prompt and saved in the indicated directory
    """

    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    #Which statistics are selected with the two scores?
    selected_columns_board = board_df.columns[board_df.loc['ERA5_1981/2010'] >= thre_ERA5].values
    selected_columns_selrate = selrate_df.columns[selrate_df.loc['ERA5_1981/2010'] >= thre_ERA5].values

    #Which clusters do they correspond to?
    selected_clusters_board = list(set([col.rsplit('_', 1)[0] for col in selected_columns_board]))
    selected_clusters_selrate = list(set([col.rsplit('_', 1)[0] for col in selected_columns_selrate]))
    prefixes = list(set(selected_clusters_board).union(selected_clusters_selrate))
    
    
    # Select columns that start with any of the prefixes, retrieve to which clusters are they related
    # for loops to keep the order
    selected_columns = []
    for col in board_df.columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected_columns.append(col)    
    
    benchmark_drivers = []
    seen = set()
    for col in selected_columns:
        prefix = col.rsplit('_', 1)[0]
        if prefix not in seen:
            seen.add(prefix)
            benchmark_drivers.append(prefix)
        
    # Subset the DataFrame
    filtered_df = board_df[selected_columns]     
    board_filled = filtered_df.fillna(-9)

    selrate_filled = selrate_df[selected_columns]
    
    # Define the colors for the custom colormap   
    cmap_colors = [(0, 'white'),
              (thre_CMIP6-tolerance,'white'),
              (thre_CMIP6,(0.6241, 0.7809, 0.8855)),
              ((1+thre_CMIP6-tolerance)/2, (0.1137, 0.4902, 0.7608)),
              (1,(0.0157, 0.1569, 0.3412))]

    positions, colors = zip(*cmap_colors)
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))
    
    # Sample colors at 0.05, 0.15, ..., 0.95
    sample_points = np.arange(0.05, 1.0, 0.1)
    discrete_colors = custom_cmap(sample_points) 
    # Create a ListedColormap from discrete colors
    discrete_cmap = ListedColormap(discrete_colors)

    vmin, vmax = 0, 1

    # Create a mask for values outside the range
    mask = (board_filled < 0) | (board_filled > 1)

    # Create a heatmap
    plt.figure(figsize=(0.35*board_filled.shape[1], 0.35*board_filled.shape[0]))
    heatmap = sns.heatmap(board_filled, cmap=discrete_cmap, fmt=".1f", vmin=vmin, vmax=vmax, mask=mask, cbar_kws={'label': 'Values'}, cbar=False)

    # Set color for NaN values
    heatmap.set_facecolor('lightgrey')

    xticks = np.arange(1.5, len(board_filled.columns)+1.5, 3)
    #print(f'we have {len(xticks)} xticks: {xticks}')
    #heatmap.set_xticks(ticks)
    plt.xticks(xticks)
    
    # Set the updated x-tick labels for the heatmap

    fs_to_acronym = dict(zip(acronyms_drivers['fs_name'], acronyms_drivers['acronym']))
    acr_labels = pd.Series(benchmark_drivers).replace(fs_to_acronym).tolist()
    #print(benchmark_drivers)
    #print(acr_labels)
    
    
    
    heatmap.set_xticklabels(acr_labels  , ha='right', fontsize=24, rotation=45)   
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=20)

    plt.axhline(0, color='darkgreen', linewidth=3, linestyle='solid')
    plt.axhline(1, color='darkgreen', linewidth=3, linestyle='solid')

    # Add thin grid lines to separate models
    for i in range(0, len(board_filled.index), 1):
        plt.axhline(i + 1, color='black', linewidth=0.5, linestyle='solid')

    # Add thick grid lines to separate clusters
    for i in range(0, len(board_filled.columns), 3):
        plt.axvline(i, color='black', linewidth=5, linestyle='solid')
        
    # #Add thick grid lines every 15 cells to separate variables
    # for i in range(0, len(board_filled.columns)+15, 15):
    #     plt.axvline(i, color='black', linewidth=4, linestyle='solid')  
    
    ax = plt.gca()
    for i in range(len(selrate_filled)):
        for j in range(len(selrate_filled.columns)):
            if selrate_filled.iloc[i, j] == 1:
                rect = Rectangle(
                    (j, i), 1, 1, fill=False, hatch='////', edgecolor='darkgreen', linewidth=1
                )
                ax.add_patch(rect)


    #plt.title(f'Board of relevant drivers for ERA5 and CMIP6 simulations in {warming4plot}, {region} ({months_code})', fontsize = 40)
    plt.title(f'Lag-agreement board of PCRO-SL, ERA5 & CMIP6 ({warming})', fontsize = 40)
    
    # # Add colorbar with custom colormap
    # colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colorbar_cmap), ax=plt.gca(), fraction=0.05, pad=0.04)
    # colorbar.set_label('Fraction of times predictors are selected', fontsize=14)  # Adjust label as needed

    # Set label for color bar
    # cbar_label = "Fraction of times predictors are selected (negative values for ERA5 case)"
    # colorbar = heatmap.figure.colorbar(heatmap.collections[0], ax=heatmap.axes, orientation='vertical', label=cbar_label)
    # colorbar.set_label(cbar_label, fontsize=32)  # Set label with fontsize
    # colorbar.ax.tick_params(labelsize=14)


    # cbar = heatmap.colorbar(label="Number of solutions")
    # cbar.set_label("Fraction of top solutions", fontsize=24)
    # cbar.ax.tick_params(labelsize=20, rotation=0)
    # cbar.set_ticks(levels)
    # percent_labels = [f"{level/pc*100:.0f}%" for level in levels]
    # cbar.set_ticklabels(percent_labels)
    
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable
    
    # Define bin edges from 0 to 1 in 10% steps (11 boundaries for 10 bins)
    levels = np.linspace(0, 1, len(discrete_colors) + 1)
    norm = BoundaryNorm(boundaries=levels, ncolors=len(discrete_colors))
    
    # Create ScalarMappable
    sm = ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])
    
    # Add discrete colorbar
    cbar = heatmap.figure.colorbar( sm,
                                    ax=heatmap.axes,
                                    orientation='vertical',
                                    pad=0.02,
                                    fraction=0.025)  # controls thickness relative to the main plot
                                    #shrink=1 )      # shrinks the length (height for vertical bar))

    cbar.set_label("Fraction of top solutions", fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    
    # Set ticks at bin **boundaries**
    cbar.set_ticks(levels)
    
    # Set labels as integer percentages (0 to 100)
    percent_labels = [f"{int(p * 100)}%" for p in levels]
    cbar.set_ticklabels(percent_labels)

    plt.savefig(f'{plot_path}/benchmarkdriversboardCMIP6_{region}_{months_code}_{warming}', dpi=600, bbox_inches='tight', transparent=False)
    # Show the plot
    plt.show()


def board_all_drivers(board_df,selrate_df,top_pairs,tolerance,thre_ERA5,thre_CMIP6,region,months_code,warming,varspecs,plot_path):
    
    """
        Produce board of the selected drivers for ERA5 and CMIP6
        
        Parameters:
        
            board_df: pd dataframe
                scores as a result of feature selection 
            labels1: list
                custom labels for x axis
            top_pairs: list of lists of str
                selected pairs of drivers which will be highlighted in the plot
            tolerance: float
                range of acceptance below the threshold to be included anyway
            thre_ERA5: float
                threshold for a drivers to be selected in ERA5
            thre_CMIP6: float
                same as above, but for CMIp6
            region: str
                region considered (for the title of the plot)
            months_code: str
                codes of the months (initials)
            warming: str
                which warming is this plot related to?
            plot_path: str
                path where you want to save the plot
        Returns:
            plot in prompt and saved in the indicated directory
    """

    ## CUSTOMIZE LABELS FOR PLOT
    
    
    labels1=[]
    for n,name in enumerate(board_df.columns):
        parts = name.split('_')
        variable = parts[0]  # 'tasmax'
        domain = parts[1:-2]    # 'Europe'
        component = parts[-2]  # 'cllow00'
        statistic = parts[-1]  # 'mean'
        if n%15==0:
            labels1.append(name)
        elif n%3==0:
            labels1.append(f'{component[-2:]}_{statistic}')
        else:
            labels1.append(statistic)

    
    if top_pairs is not None:
        ## find the indices of the first pair in the list of drivers
        i_dr = [np.nan,np.nan]
        i_dr[0]=np.where(board_df.columns.str.startswith(top_pairs[0][0]))[0][0]
        i_dr[1]=np.where(board_df.columns.str.startswith(top_pairs[0][1]))[0][0]
    
        ## find the indices of the second pair in the list of drivers
        j_dr = np.full(len(top_pairs[1]), np.nan)
        for j in range(0,len(top_pairs[1])):
            j_dr[j]=np.where(board_df.columns.str.startswith(top_pairs[1][j]))[0][0]

 
    board_filled = board_df.fillna(-9)


    # Define the colors for the custom colormap
    colors = [(0, (50/255, 95/255, 85/255)), 
              ((1+(-1-thre_ERA5)/2+tolerance/2)/2, 'seagreen'),
              ((1-thre_ERA5+tolerance)/2, 'white'), 
              ((1+thre_CMIP6-tolerance)/2, 'white'), 
              ((1+(1+thre_CMIP6)/2-tolerance/2)/2, (0.24816608996539793, 0.5618915801614763, 0.7709803921568628)),
              (1,(0.03137254901960784, 0.21259515570934256, 0.4557785467128028))]

    # Create the custom colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors)

    vmin, vmax = -1, 1

    # Create a mask for values outside the range
    mask = (board_filled < -1) | (board_filled > 1)

    # Create a heatmap
    plt.figure(figsize=(36, 1*board_filled.shape[0]))
    heatmap = sns.heatmap(board_filled, cmap=custom_cmap, fmt=".1f", vmin=vmin, vmax=vmax, mask=mask, cbar_kws={'label': 'Values'}, cbar=False)

    # Set color for NaN values
    heatmap.set_facecolor('lightgrey')

    # # Set double height for the first row
    # ax = heatmap.axes
    # ax.set_ylim(0.5, len(board_filled) + 0.5)

    xticks = np.arange(12, len(board_filled.columns)+12, 15)

    #heatmap.set_xticks(ticks)
    plt.xticks(xticks)
    # Create a mapping from CMIP6 values to name values (in capital letters)
    cmip6_to_name = dict(zip(varspecs['CMIP6'], varspecs['name'].str.upper()))
    
    # Process the labels and replace the first part (variable) with the corresponding name from CMIP6
    updated_labels = [
        '_'.join([cmip6_to_name.get(lab.split('_')[0], lab.split('_')[0]).upper()] + lab.split('_')[1:2])
        for lab in labels1[0:len(board_filled.columns)+1:15]
    ]
    
    # Check the updated labels
    # print("Updated Labels:", updated_labels)
    
    # Set the updated x-tick labels for the heatmap
    heatmap.set_xticklabels(updated_labels, ha='right', fontsize=32, rotation=45)
    
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=32, rotation=0)

    plt.axhline(0, color='darkgreen', linewidth=3, linestyle='solid')
    plt.axhline(1, color='darkgreen', linewidth=3, linestyle='solid')

    # Add thin grid lines to separate models
    for i in range(0, len(board_filled.index), 1):
        plt.axhline(i + 1, color='black', linewidth=0.5, linestyle='solid')

    # Add thick grid lines to separate clusters
    for i in range(0, len(board_filled.columns), 3):
        plt.axvline(i, color='black', linewidth=0.5, linestyle='dashed')
        
    #Add thick grid lines every 15 cells to separate variables
    for i in range(0, len(board_filled.columns)+15, 15):
        plt.axvline(i, color='black', linewidth=4, linestyle='solid')  

    if top_pairs is not None:    
        # Plot the columns related to the two pairs
        for i in j_dr:
            plt.axvspan(i, i+3, color='gold', alpha=0.2)
            plt.axvline(i, color='darkorange', linewidth=3, linestyle='-')#(0, (3, 10, 1, 10)))
            plt.axvline(i+3, color='darkorange', linewidth=3, linestyle='-')#(0, (3, 10, 1, 10)))
    
        for i in i_dr:
            plt.axvspan(i, i+3, color='violet', alpha=0.2)
            plt.axvline(i, color='purple', linewidth=3, linestyle=(0, (5, 7)))
            plt.axvline(i+3, color='purple', linewidth=3, linestyle=(0, (5, 7)))
    
    ax = plt.gca()
    for i in range(len(selrate_df)):
        for j in range(len(selrate_df.columns)):
            if selrate_df.iloc[i, j] == 1:
                rect = Rectangle(
                    (j, i), 1, 1, fill=False, hatch='//', edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)


    #plt.title(f'Board of relevant drivers for ERA5 and CMIP6 simulations in {warming4plot}, {region} ({months_code})', fontsize = 40)
    plt.title('Lag-agreement board of PCRO-SL, ERA5 & CMIP6 (CWS14.2)', fontsize = 40)
    
    # # Add colorbar with custom colormap
    # colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=colorbar_cmap), ax=plt.gca(), fraction=0.05, pad=0.04)
    # colorbar.set_label('Fraction of times predictors are selected', fontsize=14)  # Adjust label as needed

    # Set label for color bar
    cbar_label = "Lag-agreement selection share \n (negative values for ERA5 case)"
    colorbar = heatmap.figure.colorbar(heatmap.collections[0], ax=heatmap.axes, orientation='vertical', label=cbar_label)
    colorbar.set_label(cbar_label, fontsize=32)  # Set label with fontsize
    colorbar.ax.tick_params(labelsize=20)
    
    # Create the custom hatched legend patch
    hatched_patch = Patch(
        facecolor='white',
        edgecolor='black',
        hatch='//',
        label='Selected by selection rate'
    )
    
    # Add the legend outside the heatmap in the bottom-right corner
    plt.legend(
        handles=[hatched_patch],
        loc='lower right',
        bbox_to_anchor=(1.25, -0.25),  # (x, y) relative to the axes
        fontsize=20,
        frameon=True
    )  
    
    ## Force horizontal y ticks
    plt.setp(heatmap.get_yticklabels(), rotation=0, ha='right', fontsize=32)
    
    plt.savefig(f'{plot_path}/driversboardCMIP6_{region}_{months_code}_{warming}', dpi=600, bbox_inches='tight', transparent=False)
    # Show the plot
    plt.show()

def calculate_averages(df_w, df_base, series_ERA5):

    num_mdls = df_base.shape[1]

    
    ### Average and uncertainties for each simulation
    sim_means_base = df_base.mean().to_frame().T
    sim_means_w = df_w.mean().to_frame().T
    sim_uncs_base = df_base.std().to_frame().T/np.sqrt(df_base.shape[0])        
    sim_uncs_w = df_w.std().to_frame().T/np.sqrt(df_w.shape[0])    

    ### Nested average of the simulation averages
    avg_n_base, unc_n_base = nested_mean(sim_means_base,sim_uncs_base)
    avg_n_w, unc_n_w = nested_mean(sim_means_w,sim_uncs_w)


    
    # mean_avg_n_base = np.mean(avg_n_base)
    # unc_avg_n_base = np.mean(unc_n_base)
    
    # mean_avg_n_w = np.mean(avg_n_w)
    # unc_avg_n_w = np.mean(unc_n_w)        

    increase = avg_n_w-avg_n_base #np.mean(increase_arr)
    unc_incr = unc_n_w+unc_n_base #np.mean(unc_incr_arr)

    row = np.array([np.mean(series_ERA5),np.std(series_ERA5)/np.sqrt(len(series_ERA5)),
                    avg_n_base,unc_n_base,
                    avg_n_w,unc_n_base,
                    increase,unc_incr])
    #print(row)
    row = np.round(row,2)

    return(row)   

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', linestyle='none', rotate = True, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    facecolor : str
        Facecolor of the ellipse.
    kwargs
        Additional keyword arguments passed to `matplotlib.patches.Ellipse`.

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    
    if (rotate == True):
        cov = np.cov(x, y)
        #print(cov)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # The orientation of the ellipse (i.e., the rotation of the semi-major axis)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # The lengths of the semi-major and semi-minor axes
        width, height = 2 * n_std * np.sqrt(eigenvalues)

    else:
        angle = 0
        width = 2 * n_std * np.std(x)
        height = 2 * n_std * np.std(y)

    
    
    # The ellipse
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=angle,  # Set angle to 0 for aligned axes
                      facecolor=facecolor, linestyle=linestyle, **kwargs)

    ax.add_patch(ellipse)

    return ellipse


def create_pairs_table_board_selrate(board_df,selrate_df,thre_ERA5,thre_CMIP6):
    
    """
        Given the results of lag-agreement and selection rate, find how many pair of drivers have the most validated models
        
        Parameters:
        
            board_df: pd dataframe
            thre_ERA5: float
            thre_CMIP6: float   
         
        Returns:
            pairs_table: pd dataframe
                table of the pairs of drivers with the count

    """

    # Convert to NumPy array and negate values for plotting reasons
    ERA5scores_stats = pd.DataFrame(-board_df.loc['ERA5_1981/2010'])
    ERA5notdisc = pd.to_numeric(selrate_df.loc['ERA5_1981/2010'])

    ERA5scores_stats['sel_by_selrate'] = ERA5notdisc.values
    ERA5scores_stats.rename(columns={'ERA5_1981/2010': 'board_score'}, inplace=True)
    ERA5scores_stats['cluster'] = ERA5scores_stats.index.str.rsplit('_', n=1).str[0]

    ERA5scores_stats["board_score"] = pd.to_numeric(ERA5scores_stats["board_score"], errors="coerce")
    ERA5scores = ERA5scores_stats.loc[ERA5scores_stats.groupby("cluster")["board_score"].idxmax()]
    ERA5scores['sel_by_board'] = (ERA5scores['board_score'] > thre_ERA5).astype(int)

    ERA5drivers = ERA5scores.loc[(ERA5scores.sel_by_selrate == 1) | (ERA5scores.sel_by_board == 1)]
    
    clusters = ERA5drivers.cluster.values
    order = ['tasmax', 'zg', 'psl', 'pr', 'mrsos', 'rlut', 'tos', 'siconc']  
    
    # Sort variables based on the predefined order
    clusters = sorted(clusters, key=lambda x: order.index(get_variable(x)))
    
    temp_on_site = 'tasmax_Europe_cllow00'

    pairs_table = pd.DataFrame(0,columns = clusters, index = clusters)
    pairs_table_or = pd.DataFrame(0,columns = clusters, index = clusters)


    ## For each pair, count how many models have found that pair to be relevant    
    for i,dr1 in enumerate(pairs_table.index):
        for j,dr2 in enumerate(pairs_table.columns):
            if i < j:
                for testname in board_df.index[1:]:   

                    # dr1 = pairs_table.index[4]
                    # dr2 = pairs_table.columns[9]
                    # testname = board_df.index[25]
                    # print(dr1,dr2,testname)

                    sel_by_b = False
                    sel_by_sr = False                        
                    
                    #does this driver exceed the threshold in any statistics of thes testnames?
                    dr1_bool_b = any(board_df.loc[testname,board_df.columns.str.startswith(dr1)]>thre_CMIP6) 
                    dr2_bool_b = any(board_df.loc[testname,board_df.columns.str.startswith(dr2)]>thre_CMIP6)
                    if (dr1_bool_b & dr2_bool_b):
                        #print(f'{testname} {dr1} {dr2}')
                        #print('!!!!!!!!!!!!!!!!!!!!!bingo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        sel_by_b = True
                        pairs_table.loc[dr1,dr2] = pairs_table.loc[dr1,dr2]+1   

                    #does this driver exceed the threshold in any statistics of thes testnames?
                    dr1_bool_sr  = any(selrate_df.loc[testname,selrate_df.columns.str.startswith(dr1)]==1) 
                    dr2_bool_sr = any(selrate_df.loc[testname,selrate_df.columns.str.startswith(dr2)]==1)
                    if (dr1_bool_sr & dr2_bool_sr):
                        #print(f'{testname} {dr1} {dr2}')
                        #print('!!!!!!!!!!!!!!!!!!!!!bingo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        sel_by_sr = True
                        pairs_table.loc[dr2,dr1] = pairs_table.loc[dr2,dr1]-1    

                    if sel_by_b or sel_by_sr:                        
                        pairs_table_or.loc[dr1,dr2] = pairs_table_or.loc[dr1,dr2]+1
                        pairs_table_or.loc[dr2,dr1] = pairs_table_or.loc[dr2,dr1]+1
                        
    return(pairs_table,pairs_table_or)

def get_variable(var):
    return var.split('_')[0]  # Extract prefix before the first underscore

def heatmap_pairs_table_v2(pairs_table_or, on_site, min_pop_pair):
    """
        Plot a detailed heatmap showing the number of validated CMIP6 simulations for each cluster pair.
        Highlights on-site variables, identical variable pairs, and top-performing pairs using hatches and circles.

        Parameters:
        
            pairs_table_or: pd.DataFrame
                Symmetric matrix of counts representing validated CMIP6 simulations for each cluster pair.

            on_site: list or set of str
                Names of clusters or drivers considered "on-site", which should be highlighted with hatching.

            min_pop_pair: integer
                Minimum number of validated models required to start storyline construction
                
        Returns:
            top_pairs_list: list of tuples
                List of cluster pairs (and their value) that surpass the defined threshold and are not already emphasized 
                via on-site or same-variable grouping.
    """
    # Create a mask for the lower triangle
    mask = np.tril(np.ones_like(pairs_table_or, dtype=bool),k=-1)
    top_pairs_list = []
    
    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=(0.6*pairs_table_or.shape[0], 0.6*pairs_table_or.shape[0]))  # Corrected line
    ax = sns.heatmap(pairs_table_or, mask=mask, annot=pairs_table_or, cmap='Blues', yticklabels=True,cbar=False)  # Move legend to bottom)
    
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_position([0.2, 0.93, 0.6, 0.03])  # (left, bottom, width, height)
    
    plt.gca().xaxis.set_ticks_position('top')  # Move x-axis labels to the top
    plt.xticks(rotation=30, ha='left')  # Rotate x-axis labels by 30 degrees
    
    # Add diagonal hatch shading with dark grey patches
    for i in range(pairs_table_or.shape[0]):  
        for j in range(pairs_table_or.shape[1]):  
            if i<=j:
                row_label = pairs_table_or.index[i]
                col_label = pairs_table_or.columns[j]
        
                if row_label in on_site or col_label in on_site:
                    hatch_style = '////'  # Diagonal shading for on_site rows/columns
                    clr = 'darkgrey'
                    alpha=1
                else:
                    continue  
        
                # Add dark grey patches with line thickness 0.3
                rect = Rectangle(
                    (j, i), 1, 1, fill=False, hatch=hatch_style,
                    edgecolor=clr, alpha=alpha, linewidth=0, transform=ax.transData, clip_on=False
                )
                ax.add_patch(rect)
    for i in range(pairs_table_or.shape[0]):  
        for j in range(pairs_table_or.shape[1]):  
            if i<=j:
                row_label = pairs_table_or.index[i]
                col_label = pairs_table_or.columns[j]
            
                if row_label.startswith('TP') and col_label.startswith('TP'):
                    hatch_style = '\\' 
                    clr = 'darkgrey'
                    alpha=0.8
                elif ((row_label.startswith('MSLP') or row_label.startswith('Z500')) and 
                     (col_label.startswith('MSLP') or col_label.startswith('Z500'))):
                    hatch_style = '\\'  
                    clr = 'darkgrey'
                    alpha=0.8
                elif row_label.startswith('TMAX') and col_label.startswith('TMAX'):
                    hatch_style = '\\'  
                    clr = 'darkgrey'
                    alpha=0.8
                elif row_label.startswith('SST') and col_label.startswith('SST'):
                    hatch_style = '\\'  
                    clr = 'darkgrey'
                    alpha=0.8
                else:
                    continue  
                # Add dark grey patches with line thickness 0.3
                rect = Rectangle(
                    (j, i), 1, 1, fill=False, hatch=hatch_style,
                    edgecolor=clr, alpha=alpha, linewidth=0, transform=ax.transData, clip_on=False
                )
                ax.add_patch(rect)
    # Add circles for the specific condition (MSLP-NOS and TP-NAB)
    for i in range(pairs_table_or.shape[0]):  
        for j in range(pairs_table_or.shape[1]): 
            if i<=j:
                row_label = pairs_table_or.index[i]
                col_label = pairs_table_or.columns[j]
                value = pairs_table_or.iloc[i, j]  # Get the value in the cell
        
                # Check if the value is greater than 10 and doesn't match previous conditions
                if (
                    not (row_label in on_site or col_label in on_site)
                    and not (row_label.startswith('TP') and col_label.startswith('TP'))
                    and not ((row_label.startswith('MSLP') or row_label.startswith('Z500')) and 
                             (col_label.startswith('MSLP') or col_label.startswith('Z500')))
                    and not (row_label.startswith('TMAX') and col_label.startswith('TMAX'))
                    and not (row_label.startswith('SST') and col_label.startswith('SST'))
                    and value >= min_pop_pair
                ):
                    # Coordinates of the cell center (add 0.5 to position at the center)
                    circle = Circle((j + 0.5, i + 0.5), radius=0.4, edgecolor='darkslategray', facecolor='none', linewidth=2)
                    ax.add_patch(circle)
                    if i<j:
                        top_pairs_list.append((row_label, col_label, value))
    
    # Move y-axis labels to the right
    ax.yaxis.tick_right()
    
    # Adjust y-ticks to align properly
    ax.set_yticks(np.arange(len(pairs_table_or)) + 0.5)
    ax.set_yticklabels(pairs_table_or.index, rotation=0, ha='left', fontsize=10)
    
    
    # Ensure correct layout
    plt.tight_layout()
    # Ensure patches align properly
    ax.set_aspect('equal')
    fig.canvas.draw()        
    
    plt.title('Number of validated CMIP6 simulations \n for each cluster pair (excluding on-site drivers)')
    plt.show()
    #return(top_pairs_list)

def identify_statistics (var_clus, board_sel, selrate_sel, thre_CMIP6_sel, thre_CMIP6_low):
    
    """
        Selection of the statistics (mean, pc25, pc75) that has the best scores
        
        Parameters:
            var_clus: str
                name of the predictor without the statistic (variable_domain_cluster)
            board_sel: pd dataframe
                board of scores
            selrate_sel: pd dataframe
                board of not-discarded 
            thre_CMIP6_sel: float
                at least one statistic should exceed this threshold
            thre_CMIP6_low: float
                the selected statistic should exceed this threshols
        Returns:
            stat_sel_df: pd dataframe
                scores of the selected statistics
            selected_predictor: str
                name of the selected predictor
    """

    stat_board_df = board_sel.loc[:,board_sel.columns.str.startswith(var_clus)] #select the columns
    stat_notdisc_df = selrate_sel.loc[:,selrate_sel.columns.str.startswith(var_clus)]
    stat_sel_df = stat_board_df[(stat_board_df.max(axis=1) > thre_CMIP6_sel) | (stat_notdisc_df.max(axis=1)==1)] #only those models for which at least one stat exceeds 0.5

    avg_row = stat_sel_df.mean(axis=0)
    avg_row_df = pd.DataFrame(avg_row).T
    avg_row_df.index = ['scores_mean']
    
    count_row_df = stat_sel_df.gt(thre_CMIP6_low).sum()  # Count models exceeding threshold
    count_row_df = pd.DataFrame(count_row_df).T
    count_row_df.index = ['count_sel']


    stat_sel_df = pd.concat([stat_sel_df, avg_row_df,count_row_df]).applymap(pd.to_numeric, errors='coerce')
    stat_sel_df.loc['composite_score'] = stat_sel_df.loc[['scores_mean','count_sel']].sum()
    selected_predictor = stat_sel_df.loc['composite_score'].idxmax()
    #print(first_stat_sel_df)
    return (stat_sel_df,selected_predictor)
    
def identify_validated_pairs_list(pairs_table_or, on_site, replace_cl_names,min_pop_pairs):
    """
    Create a ranked list of validated driver pairs from a matrix of validation counts, 
    excluding on-site drivers and identical variable groups.

    Parameters:
        pairs_table_or: pd.DataFrame
            A symmetric DataFrame containing the number of validated models for each pair of clusters or drivers.

        on_site: list or set of str
            List of driver names to be excluded from the pair combinations (e.g., on-site or reference drivers).

        replace_cl_names: dict
            Dictionary mapping original cluster names to their aliases or display names. Used for replacing
            names in the final list of driver pairs.

        min_pop_pairs: int
            Minimum number of pairs to start the construction

    Returns:
        top_pairs: pd.DataFrame
            A DataFrame sorted by the number of validated models, listing driver pairs and their validation counts,
            with cluster names replaced by their original identifiers using the inverse of `replace_cl_names`.
    """
    
    
    
    # Remove columns with only zeros
    #pairs_table = pairs_table.loc[:, (pairs_table != 0).any(axis=0)]
    # Remove rows with only zeros
    #pairs_table = pairs_table[(pairs_table != 0).any(axis=1)]    

    pairs_table1 = pairs_table_or.drop(index=on_site, columns=on_site)    

    for i in range(pairs_table1.shape[0]):  
        for j in range(pairs_table1.shape[1]):  
            row_label = pairs_table1.index[i]
            col_label = pairs_table1.columns[j]
        
            if ((row_label.startswith('TP') and col_label.startswith('TP')) or 
                    ((row_label.startswith('MSLP') or row_label.startswith('Z500')) and 
                         (col_label.startswith('MSLP') or col_label.startswith('Z500'))) or
                    (row_label.startswith('TMAX') and col_label.startswith('TMAX')) or
                    (row_label.startswith('SST') and col_label.startswith('SST'))):
                pairs_table1.loc[row_label,col_label] = 0


    # Create an empty list to store the result
    pairs_list = []
    
    # Iterate over the upper triangle of the DataFrame
    for i in range(len(pairs_table1.columns)):
        for j in range(i+1, len(pairs_table1.columns)):
            row_name = pairs_table1.columns[i]
            col_name = pairs_table1.columns[j]
            value = pairs_table1.iloc[i, j]
            pairs_list.append([row_name, col_name, value])
    
    top_pairs = pd.DataFrame(pairs_list, columns=['driver1', 'driver2', 'counts'])
    
    # Sort the DataFrame by the 'count' column
    top_pairs = top_pairs.sort_values(by='counts', ascending=False).reset_index(drop=True)

    top_pairs = top_pairs.loc[top_pairs.counts>=min_pop_pairs]
    
    # Reverse the dictionary to map values back to keys
    reverse_replace_cl_names = {v: k for k, v in replace_cl_names.items()}
    
    # Replace the values in 'driver1' and 'driver2' with the keys from the reversed dictionary
    top_pairs['driver1'] = top_pairs['driver1'].replace(reverse_replace_cl_names)
    top_pairs['driver2'] = top_pairs['driver2'].replace(reverse_replace_cl_names)


    return(top_pairs)
    
def is_point_inside_ellipse(x, y, ellipse):
    """
    Check if a point (x, y) is inside the ellipse defined by its parameters.
    
    Parameters:
        x (float): x-coordinate of the point
        y (float): y-coordinate of the point
        ellipse (matplotlib.patches.Ellipse): Ellipse object representing the ellipse
    
    Returns:
        bool: True if the point is inside the ellipse, False otherwise
    """
    # Get ellipse properties
    center_x, center_y = ellipse.center
    width, height = ellipse.width, ellipse.height
    angle = np.radians(ellipse.angle)

    # Calculate the point relative to the center of the ellipse
    dx = x - center_x
    dy = y - center_y

    # Apply rotation transformation
    x_rot = dx * np.cos(-angle) - dy * np.sin(-angle)
    y_rot = dx * np.sin(-angle) + dy * np.cos(-angle)

    # Check if the point is inside the ellipse equation
    is_inside = ((x_rot / (width / 2))**2 + (y_rot / (height / 2))**2) <= 1

    return is_inside

def nested_mean (means, uncs):
    
    """
    Compute a nested multi-level mean and uncertainty across climate models. One value for each simulation is expected.

    This function performs hierarchical averaging over a set of climate model outputs. 
    It is designed to handle input data structured with columns named as 
    '<model>_<member>_<scenario>'.

    Averaging steps:
        1. **Member-level averaging**: Average all available scenario values for each member.
        2. **Model-level averaging**: Average the previously computed member-level means for each model.
        3. **Ensemble-level averaging**: Average across all model-level means to obtain the final value.
    
    The same nesting is applied to the uncertainty values (typically standard errors), 
    averaging them at each stage.

    Parameters:
        means : pandas.DataFrame
            DataFrame of mean values, where each column corresponds to a unique 
            model-member-scenario combination and each row is a single entry (e.g., time or region).

        uncs : pandas.DataFrame
            DataFrame of uncertainty values (e.g., standard error) corresponding to `means`, 
            with the same shape and column structure.

    Returns:
        avg : float
            Final scalar average computed across all models, members, and scenarios.

        unc : float
            Final scalar uncertainty, averaged across all nested levels.
    """
    
    models = [fn.split('_')[0] for fn in np.array(means.columns)]
    members = [fn.split('_')[1] for fn in np.array(means.columns)]
    ssps = [fn.split('_')[2] for fn in np.array(means.columns)]
    avg_by_mdl = pd.DataFrame(columns=np.unique(models),index=means.index, data=np.nan)
    unc_by_mdl = pd.DataFrame(columns=np.unique(models),index=uncs.index, data=np.nan)
    for sub_mdl in np.unique(models):
        sub_means_mdl = means.loc[:,[m == sub_mdl for m in models]]
        sub_uncs_mdl = uncs.loc[:,[m == sub_mdl for m in models]]
        sub_mmb = np.array(members)[np.array([model == sub_mdl for model in models])] ## list of member names for this model
        avg_by_mmb = pd.DataFrame(columns=np.unique(sub_mmb),index=sub_means_mdl.index, data=np.nan)
        unc_by_mmb = pd.DataFrame(columns=np.unique(sub_mmb),index=sub_uncs_mdl.index, data=np.nan)
        for mmb in np.unique(sub_mmb):
            sub_means_mdl_mmb = sub_means_mdl.loc[:,[m == mmb for m in sub_mmb]]
            sub_uncs_mdl_mmb = sub_uncs_mdl.loc[:,[m == mmb for m in sub_mmb]]            
            avg_by_mmb[mmb] = sub_means_mdl_mmb.mean(axis=1)
            unc_by_mmb[mmb] = sub_uncs_mdl_mmb.mean(axis=1)
        avg_by_mdl[sub_mdl] = avg_by_mmb.mean(axis=1)
        unc_by_mdl[sub_mdl] = unc_by_mmb.mean(axis=1)
    avg = avg_by_mdl.mean(axis=1).values[0]
    unc = unc_by_mdl.mean(axis=1).values[0]
    return(avg,unc)     

def nested_mean_30values(df):

    """
    Compute a nested average and uncertainty over series of data related to the simulations.

    This function processes a DataFrame containing values from multiple climate models, 
    where each column is expected to follow a naming convention like: 
    '<model>_<member>_<scenario>' (e.g., 'EC-Earth3_mmb01_ssp245').

    The nested averaging proceeds as follows:
        1. **Member-level averaging:** Average over scenarios within each member.
        2. **Model-level averaging:** Average across members within each model.
        3. **Multi-model averaging:** Average across different models.
    
    Along with the final average, it estimates uncertainty using the standard error of the mean.

    Parameters:
        df : pandas.DataFrame
            DataFrame where columns are named by model_member_scenario 
            and rows correspond to time or spatial entries.

    Returns:
        avg : pandas.Series
            Series of multi-model average values (row-wise).
        
        unc : pandas.Series
            Series of uncertainty estimates based on the standard deviation 
            across all values in the input DataFrame, divided by sqrt(N).
            (Note: Final `unc` is simplified as an overall uncertainty, 
            not fully nested across members/models.)
    """
    #print(df.columns)
    
    models = [fn.split('_')[0] for fn in np.array(df.columns)]
    members = [fn.split('_')[1] for fn in np.array(df.columns)]
    ssps = [fn.split('_')[2] for fn in np.array(df.columns)]
    avg_by_mdl = pd.DataFrame(columns=np.unique(models),index=df.index, data=np.nan)
    unc_by_mdl = pd.DataFrame(columns=np.unique(models),index=df.index, data=np.nan)
    
    sub_mdl = models[0]
    for sub_mdl in np.unique(models):
        #print(mdl)
        sub_df_mdl = df.loc[:,[m == sub_mdl for m in models]]
        sub_mmb = np.array(members)[np.array([model == sub_mdl for model in models])] ## list of member names for this model
        avg_by_mmb = pd.DataFrame(columns=np.unique(sub_mmb),index=sub_df_mdl.index, data=np.nan)
        unc_by_mmb = pd.DataFrame(columns=np.unique(sub_mmb),index=sub_df_mdl.index, data=np.nan)
        #mmb = 'mmb01'
        for mmb in np.unique(sub_mmb):
            #print(mmb)
            #print(sub_mmb)
            #print(sub_df_mdl.columns)
            sub_df_mdl_mmb = sub_df_mdl.loc[:,[m == mmb for m in sub_mmb]]
            avg_by_mmb[mmb] = sub_df_mdl_mmb.mean(axis=1)
            unc_by_mmb[mmb] = [np.std(sub_df_mdl_mmb.loc[y,].values)/np.sqrt(sub_df_mdl_mmb.shape[0]) for y in range(0,sub_df_mdl_mmb.shape[0])]
        avg_by_mdl[sub_mdl] = avg_by_mmb.mean(axis=1)
        unc_by_mdl[sub_mdl] = unc_by_mmb.mean(axis=1)
    avg = avg_by_mdl.mean(axis=1)
    unc = unc_by_mdl.mean(axis=1)
    
    unc = df.std(axis=1)/np.sqrt(df.shape[1])
    
    return(avg,unc)

def points_in_ellipse_and_quadrants(diff_avg, ellipse_out, ellipse_in, center_x, center_y):
    """
    Check which points in the DataFrame are inside the given ellipses and in which quadrant they lie.
    
    Parameters:
        diff_avg (DataFrame): DataFrame containing the points to check
        ellipse_out (Ellipse): Outer ellipse
        ellipse_in (Ellipse): Inner ellipse
        center_x (float): x-coordinate of the center of the ellipses
        center_y (float): y-coordinate of the center of the ellipses
    
    Returns:
        Tuple: A tuple containing lists of points inside the ellipses and in each quadrant
    """
    points_in_ellipse = []
    points_in_quadrant1 = []
    points_in_quadrant2 = []
    points_in_quadrant3 = []
    points_in_quadrant4 = []
    
    driver0 = diff_avg.columns[0]
    driver1 = diff_avg.columns[1]
    
    
    for index, row in diff_avg.iterrows():
        x = row[driver0]  
        y = row[driver1] 
        #print(index)
        # Check if the point is within the inner ellipse
        if not is_point_inside_ellipse(x, y, ellipse_in):
            #print('not in the inner')
            if is_point_inside_ellipse(x, y, ellipse_out): 
                #print('within the outer')
                points_in_ellipse.append(index)

                # Determine the quadrant
                if x > center_x and y > center_y:
                    points_in_quadrant1.append(index)
                elif x < center_x and y > center_y:
                    points_in_quadrant2.append(index)
                elif x < center_x and y < center_y:
                    points_in_quadrant3.append(index)
                elif x > center_x and y < center_y:
                    points_in_quadrant4.append(index)

    return points_in_ellipse, points_in_quadrant1, points_in_quadrant2, points_in_quadrant3, points_in_quadrant4

def resample_30_900 (series):
    series900 = np.random.choice(series.values, size=900, replace=True)
    q30 = np.quantile(series900, np.linspace(0, 1, 30))
    return(q30)


def scatter_dr_avg(diff_avg,drivers,wmg,wmg_base,num_sl,varspecs,conf_lev,rotate=True,savefig=False,transparent=False,more='',xlabel=None,ylabel=None,num_in_title=True, outpath='./'):
    """
    Create a 2D scatter plot of two climate/environmental drivers for a specified CWS,
    showing their deviation from a baseline CWS. Add statistical confidence ellipses, quadrant annotations,
    and color/marker encodings for CMIP6 model-member-scenario combinations.

    Parameters:
        diff_avg: pd.DataFrame
            DataFrame containing differences of driver averages between `wmg` and `wmg_base`.
            The DataFrame index should follow the format: "model_member_scenario".
        drivers: list of str
            A list with two strings specifying the drivers to be plotted on the x and y axes, respectively.
        wmg: str
            The target warming (e.g., 'CWS15') whose average differences are analyzed.
        wmg_base: str
            The baseline warming (e.g., 'CWS142') used for calculating deviations.
        num_sl: int
            Code of the storyline you want to plot
        varspecs: pd.DataFrame
            A DataFrame containing metadata for variables (e.g., variable names and units).
            Must include a column 'CMIP6' for matching with driver names and 'unity' for units.
        conf_lev: float
            Confidence level for the outer statistical ellipse (e.g., 0.95 for 95% confidence).
        rotate: bool, default=True
            If True, rotates the ellipses according to the data orientation. If False, keeps axes aligned.
        savefig: bool, default=False
            If True, saves the figure to disk using global variables `sl_path`, `exp_code`, `region`, etc.
        transparent: bool, default=False
            If True, saves the figure with a transparent background.
        more: str, default=''
            Additional label appended to axis titles. If starts with 'st', axis units are omitted.
        xlabel: str, optional
            Custom label for the x-axis. If None, a default label is constructed.
        ylabel: str, optional
            Custom label for the y-axis. If None, a default label is constructed.
        num_in_title: bool, default=True
            If True, includes the storyline number in the plot title. Requires global variable `num_sl`.
        outpath: str
            output path
            
    Returns:
        points_in_ellipse: list
            List of indices of data points that lie within the outer confidence ellipse.
        points_in_quadrant1: list
            List of indices of data points in the top-right quadrant (high-high).
        points_in_quadrant2: list
            List of indices of data points in the top-left quadrant (low-high).
        points_in_quadrant3: list
            List of indices of data points in the bottom-left quadrant (low-low).
        points_in_quadrant4: list
            List of indices of data points in the bottom-right quadrant (high-low).
    """    
    
    ### Get extremes and statistics of the axes
    
    x_max = (diff_avg[f'{wmg}-{wmg_base}_{drivers[0]}']).max()
    y_max = (diff_avg[f'{wmg}-{wmg_base}_{drivers[1]}']).max()
    x_min = (diff_avg[f'{wmg}-{wmg_base}_{drivers[0]}']).min()
    y_min = (diff_avg[f'{wmg}-{wmg_base}_{drivers[1]}']).min()

    x_std = (diff_avg[f'{wmg}-{wmg_base}_{drivers[0]}']).std()
    y_std = (diff_avg[f'{wmg}-{wmg_base}_{drivers[1]}']).std()

    # Calculate plot limits
    x_min = x_min - x_std
    x_max = x_max + x_std
    y_min = y_min - y_std
    y_max = y_max + y_std
    
    

    ## Order models so that the one with most members gets blue, then red, green, purple and finally orange
    # Extracting models
    models = [d.split('_')[0] for d in diff_avg.index]
    # Getting unique models
    models_unique = np.unique(models)
    # Counting occurrences of each unique model in models
    model_counts = Counter(models)
    # Reordering models_unique according to the count
    models = sorted(models_unique, key=lambda x: model_counts[x], reverse=True)
    
    members = np.unique([d.split('_')[1] for d in diff_avg.index])

    #Rearrange palette to move orange to fifth position
    first_palette = plt.cm.tab10(np.array(range(0,10)))
    second_row = first_palette[1, :]
    first_palette = np.delete(first_palette, 1, axis=0)
    first_palette = np.insert(first_palette, 4, second_row, axis=0)
    
    if len(models)<=10:
        colors = first_palette[:len(models)]
    else:
        colors0 = first_palette
        colors1 = plt.cm.Set1(np.array(range(0,len(models)-10)))
        colors = np.concatenate((colors0,colors1))

    clr_mdls = {}
    for i,model in enumerate(models):
        temp_clrs1 = [sns.light_palette(colors[i],n_colors=7)[j] for j in [1,4]]
        temp_clrs2 = [sns.dark_palette(colors[i],n_colors=7)[j] for j in [-1,-4]]
        clr_mdls[model] = np.concatenate([temp_clrs1,temp_clrs2])

    
    # Markers for different members of same model
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h',
              'o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h',
              'o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h',
              'o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'h']
    mrk_mmbs = {}
    for i,member in enumerate(members):
        mrk_mmbs[member] = markers[i]
    


    # From lighter to darker for different scenarios   
    shades_ssp = {'ssp126':0, 'ssp245':1, 'ssp370':2, 'ssp585':3}    




    plt.figure(figsize=(24, 18))
    ax = plt.gca()
    handles = []
    labels = []
    

    # Add confidence ellipse
    ellipse_out = confidence_ellipse(diff_avg[f'{wmg}-{wmg_base}_{drivers[0]}'],
                       diff_avg[f'{wmg}-{wmg_base}_{drivers[1]}'], ax,
                       n_std=np.sqrt(chi2.ppf(conf_lev, 2)), rotate = rotate, edgecolor='black', linestyle='dashed', alpha=1)
    # Add confidence ellipse
    ellipse_in = confidence_ellipse(diff_avg[f'{wmg}-{wmg_base}_{drivers[0]}'],
                       diff_avg[f'{wmg}-{wmg_base}_{drivers[1]}'], ax,
                       n_std=np.sqrt(chi2.ppf(0.05, 2)), rotate = rotate, edgecolor='black', linestyle='solid', alpha=1)    
 
    # Get ellipse properties
    #ellipse = confidence_ellipse(x, y, ax)
    center_x, center_y = ellipse_out.center
    width, height = ellipse_out.width, ellipse_out.height
    angle = ellipse_out.angle
    
    
    print(f'Center:[{np.round(center_x,2)},{np.round(center_y,2)}].\n Width {np.round(width)}. Heigth {np.round(height)}. Angle {np.round(angle)}')
    
    plt.scatter(center_x,center_y,c='black',s=20)
    
    plt.axvline(center_x, c='black', linewidth=1)
    plt.axhline(center_y, c='black', linewidth=1)    
    
    # Calculate the endpoints of the major and minor axes
    major_axis_x_right = center_x + width/2 * np.cos(np.radians(angle))
    major_axis_y_right = center_y + width/2 * np.sin(np.radians(angle))
    plt.scatter(major_axis_x_right,major_axis_y_right,c='grey',s=10)
    plt.text(major_axis_x_right+x_std/10,major_axis_y_right,f'conf lev:{conf_lev}',fontsize=9,alpha=0.25,c='black',ha='left',va='bottom')
    major_axis_x_left = center_x + width / 2 * np.cos(np.radians(180+angle))
    major_axis_y_left = center_y + width / 2 * np.sin(np.radians(180+angle))
    plt.scatter(major_axis_x_left,major_axis_y_left,c='grey',s=10)
    plt.plot([major_axis_x_right, major_axis_x_left], [major_axis_y_right, major_axis_y_left], 
             linestyle='dotted', color='black', linewidth=0.5)    
    
    minor_axis_x_top = center_x + height / 2 * np.cos(np.radians(90+angle))
    minor_axis_y_top = center_y + height / 2 * np.sin(np.radians(90+angle))
    plt.scatter(minor_axis_x_top,minor_axis_y_top,c='grey',s=10)
    minor_axis_x_btm = center_x + height / 2 * np.cos(np.radians(270+angle))
    minor_axis_y_btm = center_y + height / 2 * np.sin(np.radians(270+angle))
    plt.scatter(minor_axis_x_btm,minor_axis_y_btm,c='grey',s=10)
    plt.plot([minor_axis_x_top, minor_axis_x_btm], [minor_axis_y_top, minor_axis_y_btm], 
             linestyle='dotted', color='black', linewidth=0.5)    

    model0 = 'something' #define this so that marker can be changed when member changes and all mmb01 won't have the same markes
    member0 = 'something'
    m=0
    for i, row in enumerate(diff_avg.iterrows()):       
        index, values = row        
        model = index.split('_')[0]
        member = index.split('_')[1]
        if ((model!=model0) | (member!=member0)):
            m = m+1
            model0 = model
            member0 = member        
        ssp = index.split('_')[2]
        issp = shades_ssp[ssp]
        #print(model)
        handle = plt.scatter(values[f'{wmg}-{wmg_base}_{drivers[0]}'], 
                             values[f'{wmg}-{wmg_base}_{drivers[1]}'], 
                             color=clr_mdls[model][issp], marker=markers[m], label=index, s=200)
        handles.append(handle)
        labels.append(index)

    
    # Sort legend labels alphabetically
    sorted_labels, sorted_handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0]))

    if not(more.startswith('st')):
        unit0 = varspecs.loc[varspecs.CMIP6==drivers[0].split('_')[0],'unity'].values[0]
        unit1 = varspecs.loc[varspecs.CMIP6==drivers[1].split('_')[0],'unity'].values[0]
    else:
        unit0 = ' '
        unit1 = ' '
    if not num_in_title:
        num_sl_4plot = ''
    else:
        num_sl_4plot = num_sl
    # Add labels and title
    
    if xlabel == None:
        plt.xlabel(f'{drivers[0]} avg in {wmg}, dev. from {wmg_base} {more} [{unit0}]',fontsize=20)
    else:
        plt.xlabel(xlabel,fontsize=18)
        
    if ylabel == None:
        plt.ylabel(f'{drivers[1]} avg in {wmg}, dev. from {wmg_base} {more} [{unit1}]',fontsize=20)
    else:
        plt.ylabel(ylabel,fontsize=18)
        
    
    #plt.title(f'Storyline {num_sl_4plot} {more} scatterplot and ellipse for {wmg}, {region} ({months_code}) ',fontsize=24)
    plt.title(f"Scatterplot and ellipse of drivers' evolution between {wmg_base} and {wmg}",fontsize=32)
    
    ratio = ((x_max-x_min)/(y_max-y_min))
    print(ratio)
    if (ratio > 1/3) & (ratio < 3):
        plt.gca().set_aspect('equal', adjustable='box')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Set plot limits
    plt.text(center_x+x_std/1.8,center_y+y_std/1.8,'HH',fontsize=60,alpha=0.25,c='grey',ha='center',va='center')
    plt.text(center_x-x_std/1.8,center_y+y_std/1.8,'LH',fontsize=60,alpha=0.25,c='grey',ha='center',va='center')
    plt.text(center_x-x_std/1.8,center_y-y_std/1.8,'LL',fontsize=60,alpha=0.25,c='grey',ha='center',va='center')
    plt.text(center_x+x_std/1.8,center_y-y_std/1.8,'HL',fontsize=60,alpha=0.25,c='grey',ha='center',va='center')
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.1)
    if len(sorted_labels)<=50:
        ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(1, 0.5),fontsize=18)
    else:
        ax.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2,fontsize=18)



    # Show plot
    #plt.show()
    if savefig:
        plt.savefig(f'{outpath}/scatter_storyline_{str(num_sl).zfill(2)}{more}', dpi=300, bbox_inches='tight', transparent=transparent)


    # Call the function to get points within the ellipse and quadrants
    points_in_ellipse, points_in_quadrant1, points_in_quadrant2, points_in_quadrant3, points_in_quadrant4 = points_in_ellipse_and_quadrants(diff_avg, ellipse_out, ellipse_in, center_x, center_y)
    return(points_in_ellipse, points_in_quadrant1, points_in_quadrant2, points_in_quadrant3, points_in_quadrant4)


def storylines_diffs_boxplot(sl_code, HWind, values, averages, keys, drivers, wmg_base, wmg, unit='', output_path='./', exp_code='', region='', months_code='', poster=False):

    """
    Creates and saves a boxplot for storyline analysis. In particular it plots the distribution of projected changes.

    This function visualizes the distribution of scenario-based values across different storylines.
    It overlays custom median values and highlights relevant drivers contributing to changes
    in a climate-related indicator (e.g., heatwave intensity).

    Parameters:
        sl_code : int or str
            Identifier code for the storyline being plotted (used in title and filename).

        HWind : str
            Name of the heatwave indicator or variable being analyzed (e.g., "TXx", "HWMId").

        values : list of list-like
            Data to be plotted as boxplots. Each inner list corresponds to a storyline's values.

        averages : dict
            Dictionary mapping each storyline key to its custom median value.

        keys : list of str
            Labels for each boxplot, corresponding to storylines or configurations.

        drivers : list of str
            Names of the two dominant drivers associated with the storyline (used in the title).

        wmg_base : float or str
            The warming level baseline (e.g., 1.0°C).

        wmg : float or str
            The target warming level (e.g., 2.0°C).

        unit : str, optional
            Unit of the indicator being plotted (e.g., "°C", "%"). Default is empty.

        output_path : str, optional
            Directory where the resulting plot image will be saved. Default is current directory.

        exp_code : str, optional
            Experiment or scenario code (not currently used in filename but may be useful for extensions).

        region : str, optional
            Name of the region under analysis (used in filename).

        months_code : str, optional
            Abbreviation for the relevant months (used in filename).

        poster : bool, optional
            If True, the figure is formatted for poster-size output with additional customization.

    Behavior:
        - Creates a horizontal boxplot with custom median lines and labels.
        - Customizes color and size based on the `poster` flag.
        - Annotates boxes with driver names and formatted median values.
        - Saves the plot to a file with an informative filename, depending on `poster` mode.

    Returns:
        None
    """
    # Create boxplot
    if not poster:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, ax = plt.subplots(figsize=(24, 12))
    box = plt.boxplot(values, labels=keys, patch_artist=True, medianprops={'color': 'lightsteelblue','linewidth':0})  # Hide default median
    
    # Customize the box colors (optional)
    
    colors = ['skyblue', 'skyblue', 'skyblue', 'skyblue']
    transparent = False
    if poster:
        colors = ['skyblue', 'skyblue', 'steelblue', 'skyblue']
        transparent = True
        
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    # Plot custom medians as horizontal lines
    for i, key in enumerate(keys):
        ax.hlines(
            y=averages[key],  # Custom median value
            xmin=i + 0.78,         # Left position of the line
            xmax=i + 1.22,         # Right position of the line
            colors=(30/255, 60/255, 90/255), linewidth=2.5, zorder=3
        )
        # Add median value text to the side of the box
        ax.text(
            i + 1.25,  # X position just to the right of the box
            averages[key],  # Y position of the custom median
            f'+{averages[key]:.2f}{unit}',  # Format the median value to 2 decimal places
            horizontalalignment='left',  # Center-align the text
            verticalalignment='center',    # Vertically align the text
            color=(30/255, 60/255, 90/255),  # Dark steelblue color
            fontsize=20,  # Font size
            zorder=4  # Layer above other elements
        )
    
    # Add labels inside the box above Q1
    for i, box_ in enumerate(box['boxes']):
        # Get the statistics from the boxplot
        stats = box['medians'][i].get_ydata()  # Get the y-data of the median line
        q1 = box['whiskers'][2*i].get_ydata()[0]  # 25th percentile (Q1)
        q3 = box['whiskers'][2*i+1].get_ydata()[0]  # 75th percentile (Q3)
        # Add the dictionary name inside the box above Q1
        if not poster:
            ax.text(i + 1, q1 - (stats[0] - q1) * 0.02,  # Slightly lower position above Q1
                    keys[i], ha='center', va='bottom', fontsize=36, color=(30/255, 60/255, 90/255),alpha=0.5)
        else:
            print(stats[0] - q3)
            ax.text(i + 1, q3  + (stats[0] - q3) * 0.1,  # Slightly lower position above Q1
                    keys[i], ha='center', va='top', fontsize=36, color=(30/255, 60/255, 90/255),alpha=0.5)

    # Remove the border (spines)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    plt.xlim(0, len(keys) + 0.8)
    # Add labels and title
    plt.title(f'Storyline {sl_code}: increase of {HWind} between {wmg_base} and {wmg}°C World \n drivers: {drivers[0]}, {drivers[1]}', fontsize=16)
    plt.ylabel(f'Increase of {HWind} [{unit}]',fontsize = 18)
    #plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if not poster:
        plt.savefig(f'{output_path}boxplots{HWind}_sl_{region}_{months_code}_{wmg}_sl{str(sl_code).zfill(2)}', dpi=300, bbox_inches='tight', transparent = transparent)        
    else:
        plt.savefig(f'POSTERboxplots{HWind}_sl_{region}_{months_code}_{wmg}_sl{str(sl_code).zfill(2)}', dpi=300, bbox_inches='tight', transparent = transparent)        
        
    plt.show()


def storylines_warmings_boxplot(sl_code, HWind, values_list1, values_list2, values_benchmark, averages1, averages2, averages0, keys, drivers, wmg_base, wmg, unit='', output_path = './', exp_code='', region='', months_code='', wmg_4plot='', wmg_base_4plot=''):
    """
    Creates a side-by-side boxplot comparing the distributions of a given heatwave index 
    (or similar variable) for two climate warming scenarios (e.g., CWS15.0 vs CWS14.2),
    along with a historical benchmark (e.g., ERA5).

    This function is used for visualizing storyline-specific changes in a climate indicator 
    across different climate worlds, highlighting median changes and key drivers.

    Parameters:
        sl_code : int or str
            Identifier of the storyline being visualized (used in title and filename).

        HWind : str
            Name of the heatwave indicator or variable analyzed (e.g., "TXx", "HWMId").

        values_list1 : list of list-like
            Scenario data for the first warming configuration (e.g., CWS15.0).

        values_list2 : list of list-like
            Scenario data for the second warming configuration (e.g., CWS14.2).

        values_benchmark : list-like
            Historical benchmark data (e.g., from ERA5 for 1981–2010).

        averages1 : dict
            Dictionary of median values for each storyline in values_list1.

        averages2 : dict
            Dictionary of median values for each storyline in values_list2.

        averages0 : float
            Median value for the historical benchmark.

        keys : list of str
            Labels for each storyline configuration (used as x-axis tick labels).

        drivers : list of str
            Dominant drivers associated with this storyline (used in plot title).

        wmg_base : float or str
            Baseline warming level used in comparison (not shown on plot).

        wmg : float or str
            Target warming level used in comparison (not shown on plot).

        unit : str, optional
            Unit of the indicator (e.g., "°C", "%"). Default is empty string.

        output_path : str, optional
            Directory where the plot image will be saved. Default is current directory.

        exp_code : str, optional
            Experiment code for filename labeling. Default is empty string.

        region : str, optional
            Name of the region under analysis (used in filename).

        months_code : str, optional
            Abbreviation for the relevant months (used in filename).

        wmg_4plot : str, optional
            Custom string for the warming level label in the plot (not currently used).

        wmg_base_4plot : str, optional
            Custom string for the baseline label in the plot (not currently used).

    Behavior:
        - Plots three aligned boxplot sets (CWS15.0, CWS14.2, ERA5) for each storyline.
        - Uses colored patches and transparent box styles for visual clarity.
        - Overlays horizontal lines to indicate custom medians.
        - Annotates plots with median values and a legend.
        - Saves the figure to a file using storyline code and metadata.

    Returns:
        None
    """
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Create boxplots for the first list (green), second list (purple), and benchmark (grey)
    # Use positions to align boxplots precisely
    box1 = plt.boxplot(values_list1, labels=keys, patch_artist=True, 
                       positions=[x - 0.2 for x in range(len(keys))],  # Shifted left by 0.2
                       medianprops={'color': 'lightsteelblue', 'linewidth': 0}, 
                       whiskerprops={'alpha': 0.5},  # Only transparency
                       vert=True)
    
    box2 = plt.boxplot(values_list2, patch_artist=True, 
                       positions=[x - 0.05 for x in range(len(keys))],  # Shifted left by 0.05
                       medianprops={'color': 'lightsteelblue', 'linewidth': 0}, 
                       whiskerprops={'alpha': 0.5},  # Only transparency
                       vert=True)
    
    box0 = plt.boxplot(values_benchmark, patch_artist=True, 
                       positions=[len(keys)- 0.2],  # Positioned with same width spread
                       labels=['ERA5'],  # Add labels to match other boxplots
                       medianprops={'color': 'lightsteelblue', 'linewidth': 0}, 
                       whiskerprops={'alpha': 0.5},  # Only transparency
                       widths=0.45,
                       vert=True)
    
    # Set colors for the lists with transparency
    colors1 = ['palegreen'] * len(keys)
    colors2 = ['thistle'] * len(keys)
    colors0 = ['lightgrey'] * len(keys)
    
    # Set the colors and transparency for the boxes
    for patch, color in zip(box1['boxes'], colors1):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
    
    for patch, color in zip(box2['boxes'], colors2):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
    
    for patch, color in zip(box0['boxes'], colors0):
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
    
    # Plot custom medians for all sets of values (as horizontal lines)
    for i, key in enumerate(keys):
        # For values_list1 (green boxes)
        ax.hlines(
            y=averages1[key],
            xmin=i - 0.425,     # Adjusted to align with shifted boxplot
            xmax=i + 0.025,     # Adjusted to align with shifted boxplot
            colors='seagreen', 
            linewidth=2.5, 
            zorder=3
        )
        # For values_list2 (purple boxes)
        ax.hlines(
            y=averages2[key],
            xmin=i - 0.275,     # Adjusted to align with shifted boxplot
            xmax=i + 0.175,     # Adjusted to align with shifted boxplot
            colors='darkorchid', 
            linewidth=2.5, 
            zorder=3
        )
    # For values_benchmark (grey boxes)
    ax.hlines(
        y=averages0,
        xmin=len(keys) - 0.425,     # Adjusted to align with shifted boxplot
        xmax=len(keys) + 0.025,     # Adjusted to align with shifted boxplot
        colors='dimgrey', 
        linewidth=2.5, 
        zorder=3
    )
    
    # Add median value text next to the lines (adjusted for all)
    for i, key in enumerate(keys):
        ax.text(
            i + 0.05,  # X position for list1 (green boxes)
            averages1[key],
            f'{averages1[key]:.2f}{unit}',
            horizontalalignment='left',
            verticalalignment='center',
            color='seagreen',
            fontsize=20,
            zorder=4
        )
        
        ax.text(
            i + 0.2,  # X position for list2 (purple boxes)
            averages2[key],
            f'{averages2[key]:.2f}{unit}',
            horizontalalignment='left',
            verticalalignment='center',
            color='darkorchid',
            fontsize=20,
            zorder=4
        )
        
    ax.text(
        len(keys) + 0.05,  # X position for benchmark (grey boxes)
        averages0,
        f'{averages0:.2f}{unit}',
        horizontalalignment='left',
        verticalalignment='center',
        color='dimgrey',
        fontsize=20,
        zorder=4
    )
    
    # Create custom legend patches
    green_patch = mpatches.Patch(color='palegreen', alpha=0.6, label='CWS15.0')
    purple_patch = mpatches.Patch(color='thistle', alpha=0.6, label='CWS14.2')
    grey_patch = mpatches.Patch(color='lightgrey', alpha=0.6, label='ERA5 1981/2010')
    
    # Add legend
    plt.legend(handles=[green_patch, purple_patch,grey_patch], 
               loc='best', 
               #title='Legend', 
               fontsize=12, 
               title_fontsize=14)
    
    # Add labels and title
    #plt.title(f'Storyline {sl_code}: distributions of {HWind} between {temp_base}°C and {temp_wmg}°C World', fontsize=16)
    plt.title(f'Storyline {sl_code}: distributions of {HWind} in CWS14.2 and CWS15.0 \n drivers: {drivers[0]}, {drivers[1]}', fontsize=16)
    plt.ylabel(f'{HWind} [{unit}]', fontsize=18)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust x-axis ticks and limits
    plt.xticks([x - 0.125 for x in range(len(keys)+1)], keys + ['ERA5 1981/2010'], fontsize=14)
    plt.xlim(-0.7, len(keys) + 0.4)
    
    # Save plot to file
    plt.savefig(f'{output_path}boxplots_{HWind}_sl_{exp_code}_{region}_{months_code}_{wmg}_sl{str(sl_code).zfill(2)}', dpi=300, bbox_inches='tight', transparent=True)
    
    # Show plot
    plt.show()