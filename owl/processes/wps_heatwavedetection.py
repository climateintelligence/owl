# libraries for data analysis
from pathlib import Path
from owl.HWs_detection import *

# libraries for service perfomance
from pywps import Process, FORMATS, LiteralInput, LiteralOutput, UOM, ComplexInput, ComplexOutput
from pywps.app.Common import Metadata

# initialize logging:
import logging
LOGGER = logging.getLogger("PYWPS")

# Process discription
class HWs_detection(Process):
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

            LiteralInput('reg_name', 'Region Name', data_type='string',
                         abstract='Choose the region of your interest',
                         allowed_values=['test', 'global', 'north_pacific', 'north_atlantic', 'indian_ocean',
                                        'austral_ocean', 'tropical_atlantic', 'tropical_pacific', 'mediterranee'],
                         default='test'),

            # LiteralInput('name', 'Your name',
            #              abstract='Please enter your name.',
            #              keywords=['name', 'firstname'],
            #              data_type='string'),

            ]

        outputs = [
            ComplexOutput('output', 'netCDF containing a Heatwave index',
                          abstract='netCDF containing a Heatwave index ... and more description',
                          as_reference=True,
                          supported_formats=[FORMATS.NETCDF]),
            # ComplexOutput('plot', 'Graphical visualisation of the Heatwave',
                          # # abstract='Plot of original input file. First timestep.',
                          # as_reference=True,
                          # supported_formats=[FORMAT_PNG]),
                          ]


        super(HWs_detection, self).__init__(
            self._handler,
            identifier="HWs_detection",
            title="HWs_detection",
            version="0.1.0",
            abstract="AI-enhanced climate service to detect Heatwaves in climate datasets.",
            metadata=[
                Metadata(
                    title="HW Detection",
                    # href="https://github.com/FREVA-CLINT/duck/raw/main/docs/source/_static/crai_logo.png",
                    # role=MEDIA_ROLE),
                    # Metadata('CRAI', 'https://github.com/FREVA-CLINT/climatereconstructionAI'),
                    # Metadata('Clint Project', 'https://climateintelligence.eu/'),
                    # Metadata('HadCRUT on Wikipedia', 'https://en.wikipedia.org/wiki/HadCRUT'),
                    # Metadata('HadCRUT4', 'https://www.metoffice.gov.uk/hadobs/hadcrut4/'),
                    # Metadata('HadCRUT5', 'https://www.metoffice.gov.uk/hadobs/hadcrut5/'),
                    # Metadata('Near Surface Air Temperature',
                    #          'https://www.atlas.impact2c.eu/en/climate/temperature/?parent_id=22'),
                    )
            ],
            inputs=inputs,
            outputs=outputs,
            status_supported=True,
            store_supported=True,
        )

    def _handler(self, request, response):
        ######################################
        # import required libraries
        import numpy as np
        import xarray as xr
        from joblib import Parallel, delayed
        import joblib
        import time
        from owl.HWMI import *

        #####################################
        ### read the values of the inputs
        dataset = request.inputs['dataset'][0].file
        reg_name = request.inputs['reg_name'][0].data
        response.update_status('Prepare dataset ...', 0)
        workdir = Path(self.workdir)

        #####################################
        ### REGION OF EXPERIENCE
        if reg_name == 'north_pacific':
            lats_bnds = np.array([30,65])
            lons_bnds = np.array([120, -120])
        elif reg_name == 'north_atlantic':
            lats_bnds = np.array([30,65])
            lons_bnds = np.array([-80, 0])
        elif reg_name == 'indian_ocean':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([45, 110])
        elif reg_name == 'austral_ocean':
            lats_bnds = np.array([-90,-30])
            lons_bnds = np.array([-180, 180])
        elif reg_name == 'tropical_atlantic':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([-70, 20])
        elif reg_name == 'tropical_pacific':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([120, -70])
        elif reg_name == 'mediterranee':
            lats_bnds = np.array([30,50])
            lons_bnds = np.array([-5, 40])
        elif reg_name == 'global':
            lats_bnds = np.array([-90,90])
            lons_bnds = np.array([-180,180])
        elif reg_name == 'test':
            lats_bnds = np.array([40,50])
            lons_bnds = np.array([10,20])
        else:
            raise Exception('not regeion detected')

##################################################
### TODO: to be defined as input parameter #######

        var='t2m'
        expname='ERA5'
        memb_str='0' # there are not members for ERA5
        nrealisation=1

        reg_name ='test'

        # Region #
        if reg_name == 'test':
            min_lon = 10
            min_lat = 40
            max_lon = 20
            max_lat = 50

        if reg_name == 'Europe':
            min_lon = -15
            min_lat = 25
            max_lon = 60
            max_lat = 70
        if reg_name == 'global':
            min_lon = 0
            min_lat = -90
            max_lon = 360
            max_lat = 90

        lats_bnds = np.array([min_lat,max_lat])
        lons_bnds = np.array([min_lon, max_lon])
        # Season #
        season='15MJJA'
        season_start_day=[5,15]
        season_start_day=[8,31]

        # nday=109 #15 of May to 31st of Aug
        nday=365
        #nday = 109*3
        # Period #
        ref_year1=2015
        ref_year2=2017
        nyear=len(range(ref_year1,ref_year2))+1

        ### PERCENTILE THRESHOLD
        #percent_thresh = 95
        percent_thresh = 90
        cv='CV'
        cv_str = cv
        ### MINIMAL DURATION OF A HW
        #duration_min = 5
        duration_min = 3


        low=False # if data with 1 degree resolution is used (True) or not (False)

        if (low==True):
            # path="/work/csp/vt17420/ERA5/daily/russo/"+var+"_r360x180/"
            # Where the outputs will be saved
            # dirout="/work/csp/vt17420/ERA5/daily/russo/"+var+"_r360x180/HWMI/"
            path="../testdata/"
            dirout="../testdata/"
            lons_reg=np.arange(min_lon,max_lon,1)
            lats_reg=np.arange(min_lat,max_lat,1)
            nlon=len(lons_reg)
            nlat=len(lats_reg)
        else:
            # path="/work/csp/vt17420/ERA5/daily/global/"+var+"/"
            # dirout="/data/csp/vt17420/CLINT_PRODUCTS/"+expname+"_HWMI_"+var+"/"
            path="../testdata/"
            dirout="../testdata/"
            lons_reg=np.arange(min_lon,max_lon+0.25,0.25)
            lats_reg=np.arange(min_lat,max_lat+0.25,0.25)
            nlon=len(lons_reg)
            nlat=len(lats_reg)

        data=np.zeros((nlon,nlat,nyear,nday,1)) # daily data for all the years


        for iyear,year in enumerate(range(ref_year1,ref_year2+1)):
            days_may=np.linspace(15, 31, num=17)
            obs1=xr.open_dataset(file)
            if low==False:
            # selecting BBox:
                obs1 = obs1.sel(latitude=slice(max_lat,min_lat),longitude=slice(min_lon,max_lon))
            # selecting time and bbox
            obs=obs1.sel(time=obs1.time.time.dt.year.isin(year))
            obs = obs.sel(time=~((obs.time.dt.month == 2) & (obs.time.dt.day == 29)))

            data[:,:,iyear,:,0]=np.transpose(obs.to_array()[0,:,:,:],[2,1,0])

        nmemb=data.shape[4]
        ndayseas = nday//duration_min +1
        HWMI = np.zeros((nyear,nmemb,nlat,nlon))
        ndayexedthreshold = np.zeros((nyear,nmemb,nlat,nlon))
        subHWarray = np.zeros((nyear, ndayseas, nmemb, nlat, nlon))
        #HW = np.zeros((nyear,nmemb,nday,nlat,nlon))
        DDthreshold = np.zeros((nyear,nmemb,nlat,nlon))
        fitsubHWarray = np.zeros((nyear, ndayseas, nmemb, nlat, nlon))
        impossible_fit_list = []

        ### To be moved into owl.HMWI ?
        def parallelized_HWMIs_computation(ilat, ilon):    #, HWMI, ndayexedthreshold, DDthreshold, fitsubHWarray, subHWarray):
                HWMIyear, HWlstyear, HWstartmembyear, HWendmembyear, ndayexedthresholdyear, DDthresholdyear, subHWarrayyear, fitsubHWarrayyear, sstMeanarrayyear, impossible_fit = calc_HWMIyear(data[ilon,ilat,:,:,:], cross_valid = cv, percent_thresh = percent_thresh, duration_min = duration_min)
                #print(HWMIyear)
                HWMI[:,:,ilat,ilon]=np.array(HWMIyear)
                ndayexedthreshold[:,:,ilat,ilon]=np.array(ndayexedthresholdyear)
                subHWarray[:,:, :, ilat,ilon]=np.array(subHWarrayyear)
                DDthreshold[:,:,ilat,ilon]=np.array(DDthresholdyear)
                fitsubHWarray[:,:, :, ilat,ilon]=np.array(fitsubHWarrayyear)
                #sstMeanarray[:,:,ilat,ilon]=np.array(sstMeanarrayyear)
                impossible_fit_list.append((impossible_fit,ilat,ilon))
                return()

        # ##################################
        # ### execute the Heatwave Detection

        Parallel(n_jobs=-1, timeout = 5*3600, verbose = 20, require='sharedmem', mmap_mode='w+')(delayed(parallelized_HWMIs_computation)(ilat, ilon) for ilat in range(nlat) for ilon in range(nlon))

        # ##################################
        # ### write out the values into the output file

        if os.path.isdir(dirout)==False:
            os.makedirs(dirout)
        years=range(ref_year1,ref_year2)
        j=0
        for i,iyear in enumerate(range(ref_year1, ref_year2+1)):
            #print iyear
            parameters_str = reg_name+"_"+season+"_"+cv_str+'_percent%i'%(percent_thresh)+'_daymin%i'%(duration_min)+"_ref_"+str(ref_year1)+"_"+str(ref_year2)+"_year_"+str(iyear)
            varout1 = "HWMI"+"_"+var+"_"+parameters_str
            vout1="HWMI"+"_"+var
            fileout=dirout+varout1+".nc"#0%i01, monstart)
            print(fileout)
            if len(glob(fileout))==1:
                os.remove(fileout)
            fout=netCDF4.Dataset(fileout, "w")
            #fin=netCDF4.Dataset(targetflst[iyear])
            lat = fout.createDimension('lat', nlat)
            lon = fout.createDimension('lon', nlon)
            rea = fout.createDimension('realisation', nrealisation)
            latitudes = fout.createVariable('lat', np.float32, ('lat',))
            longitudes = fout.createVariable('lon', np.float32,  ('lon',))

            # Time variable
            timedim = fout.createDimension('time', None)
            times = fout.createVariable('time', np.float64, ('time',))
            times.units = "hours since 1970-01-01 00:00:00"
            times.calendar = 'gregorian'
            times[:]=date2num(datetime(iyear,season_start_day[0],season_start_day[1]), units=times.units,calendar = times.calendar) ####______HERE_____

            latitudes[:] = lats_reg
            lonaux = lons_reg
            longitudes[:] = lonaux
            latitudes.units = 'degree_north'
            longitudes.units = 'degree_east'

            # Create the HWMI 4-d variable
            HWMIfile = fout.createVariable(vout1, np.float32, ('time','realisation','lat','lon'))
            fout.description = 'HWMI index (Russo et al. 2014) for ' + season + ' computed in cross validation'
            fout.history = 'computed from python script by C.Prodhomme & S.Lecestre' + time.ctime(time.time())
            fout.source = 'HWMI for ' + expname
            latitudes.units = 'degree_north'
            longitudes.units = 'degree_east'
            HWMIfile.units = 'Probability'

            # Create the nb of days 4-d variable
            expercentfile = fout.createVariable("nbdaygtpercentpct_"+var, np.float32, ('time','realisation','lat','lon'))
            expercentfile.units = 'Number of days'

            # Write the HWMI variable
            HWMIaux=HWMI[i:i+1,0:1,:,:]
            HWMIfile[0:1,0:1,:,:]=HWMIaux

            # Write the number of days
            exedaux=ndayexedthreshold[i:i+1,j:j+1,:,:]
            expercentfile[0:1,0:1,:,:]=exedaux

            # Write the DDthreshold
            #DDthresholdfile=DDthreshold
            fout.close()

        ##################################
        ### set the output

        response.outputs["output"].file = dataset

        response.update_status('done.', 100)
        return response
