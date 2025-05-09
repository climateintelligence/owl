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

def board_drivers(board_df,selrate_df,top_pairs,tolerance,thre_ERA5,thre_CMIP6,region,months_code,warming,varspecs,plot_path):
    
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
    fig, ax = plt.subplots(figsize=(10, 8))  # Corrected line
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
                    and value > min_pop_pair
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
    
    plt.title('Number of validated CMIP6 simulations for each cluster pair')
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

