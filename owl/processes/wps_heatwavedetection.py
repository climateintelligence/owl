# libraries for data analysis
import numpy as np
import os
import datetime
import time
import copy
import shutil
import sys
from function_read import *
import joblib
from joblib import Parallel, delayed

# libraries for service perfomance
from pywps import Process, LiteralInput, LiteralOutput, UOM
from pywps.app.Common import Metadata

# initialize logging:
import logging
LOGGER = logging.getLogger("PYWPS")


# Process discription
class HWdetection(Process):
    """
    Process to detect heatwaves
    """

    # definition of input and output parameter
    def __init__(self):
        inputs = [
            ComplexInput('dataset', 'Add your netCDF file here',
                         abstract="Enter a URL pointing to a NetCDF file with variables where heatwave can be detected.",
                                  # "Example: "
                                  # "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/non-infilled/HadCRUT.5.0.1.0.anomalies.ensemble_mean.nc",  # noqa
                         min_occurs=1,
                         max_occurs=1,
                         supported_formats=[FORMATS.NETCDF, FORMATS.ZIP]),

            # LiteralInput('name', 'Your name',
            #              abstract='Please enter your name.',
            #              keywords=['name', 'firstname'],
            #              data_type='string')]

        outputs = [
            ComplexOutput('output', 'netCDF containing a Heatwave index',
                          abstract='netCDF containing a Heatwave index ... and more description',
                          as_reference=True,
                          supported_formats=[FORMATS.NETCDF]),
            ComplexOutput('plot', 'Graphical visualisation of the Heatwave',
                          # abstract='Plot of original input file. First timestep.',
                          as_reference=True,
                          supported_formats=[FORMAT_PNG]),
                          ]

#TODO: original code needs to be transfered to a service:
    def detectHW1year(field, lat, lon, args, allowdist=1):
        """
        field : np.array fitsubHW
        lat, lon : np.arrays corresponding to lat, lon range.
        allowdist : neighbourhood geometrical radius. The temporal radius is fixed to one.
        """
        expname, reg_name, memb_str, season, parameters_str, start_year, lats_reg, lons_reg = args
        #print("lon",lon)
        #print(lat)
        nlat= len(lat)
        nlon=len(lon)
        #print(field.shape)
        #print(field)
        HWwhere = np.ma.where(field>0) #select indices ix of field where field[ix]>0 --> There is a HW at ix
        #Maybe field>0.05 would be better?

        #print(HWwhere)
        nHWpoints = HWwhere[0].shape[0] #number of points with a HW
        #print(nHWpoints)

        if nHWpoints != 0:
            #transform HWwhere in a list of points
            HWpoint = []
            for iHW in range(nHWpoints):
                HWpoint.append((HWwhere[0][iHW],HWwhere[1][iHW], HWwhere[2][iHW]))
                # 0 --> time variable : day
                # 1,2 --> space variable : lat, lon

            #
            #_______sort heatwave points by neigbours________
            #

            HWpointaux=list(HWpoint) #make a copy
            HW = []
            iHW = 0
            iyear=0
            #initialize the list of seeds with the first point
            seedlist = [HWpointaux[0]]
            #remove seed from the list
            HWpointaux=HWpointaux[1:]
            #print seedlist
            #run over all the points
            while len(HWpointaux)>0: #still some points we did not reach

                #create a list to store the points of one HW
                ptHWlst = []
                while len(seedlist)>0:
                    #print(seedlist)
                    #remove the seed from the list of seeds and keep the current seed point
                    seedpoint=seedlist[0]
                    #print seedlist
                    #add the seed to the heatwave
                    #print ptHWlst
                    #check neighbours for spatial and temporal dimensions
                    listnei = spacelist_neighbors_highdist2(seedpoint, allowdist)

                    # adding temporal neighbours
                    neibef = (seedpoint[0]-1, seedpoint[1], seedpoint[2])

                    #if not(neibef in listnei): #&(dist>0):
                    #    listnei.append(neibef)

                    neiaft = (seedpoint[0]+1, seedpoint[1], seedpoint[2])
                    #if not(neiaft in listnei): #&(dist>0):
                    #    listnei.append(neiaft)

                    listnei = listnei+[neibef, neiaft]

                    if reg_name != "global":
                        #remove element outside the limits (avoid useless parcours of HWpointaux)
                        listnei = [nei for nei in listnei if all(0 <= x < y for x, y in zip(nei, field.shape))]
                        #need to have lats_range between 0 and 179 et lons_range between 0 and 359

                    for nei in listnei:
                        if not(nei in ptHWlst): #Not interested if neighbour has already been looked for
                            if nei in HWpointaux: #if neighbour point is indeed part of the HW
                                #add the neighbourg to the seedlist
                                seedlist.append(nei)
                                #remove the neigbourg from the heatwave list
                                HWpointaux.remove(nei)
                                #print(seedlist)
                    #add point to HW list
                    ptHWlst.append(seedpoint)
                    #
                    seedlist=seedlist[1:]

                #once the seed list is empty take a new seed it remains points
                #print HWpointaux
                if len(HWpointaux)>0:
                    seedlist = [HWpointaux[0]]
                    HWpointaux=HWpointaux[1:]
                #keep the list of point for each HW
                HW.append(ptHWlst)
        else:
            HW = []
        #
        #_______END of sort heatwave points by neigbours________
        #

        return HW
