# libraries for data analysis

from owl.HWs_detection import *

# libraries for service perfomance
from pywps import Process, LiteralInput, LiteralOutput, UOM
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
                         allowed_values=['global', 'north_pacific', 'north_atlantic', 'indian_ocean',
                                        'austral_ocean', 'tropical_atlantic', 'tropical_pacific', 'mediterranee'],
                         default='global'),

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
            ComplexOutput('plot', 'Graphical visualisation of the Heatwave',
                          # abstract='Plot of original input file. First timestep.',
                          as_reference=True,
                          supported_formats=[FORMAT_PNG]),
                          ]


        super(ClintAI, self).__init__(
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
                Metadata('Clint Project', 'https://climateintelligence.eu/'),
                # Metadata('HadCRUT on Wikipedia', 'https://en.wikipedia.org/wiki/HadCRUT'),
                # Metadata('HadCRUT4', 'https://www.metoffice.gov.uk/hadobs/hadcrut4/'),
                # Metadata('HadCRUT5', 'https://www.metoffice.gov.uk/hadobs/hadcrut5/'),
                # Metadata('Near Surface Air Temperature',
                #          'https://www.atlas.impact2c.eu/en/climate/temperature/?parent_id=22'),
            ],
            inputs=inputs,
            outputs=outputs,
            status_supported=True,
            store_supported=True,
        )

    def _handler(self, request, response):

        #####################################
        ### read the values of the inputs
        dataset = request.inputs['dataset'][0].file
        reg_name = request.inputs['reg_name'][0].data

        response.update_status('Prepare dataset ...', 0)
        workdir = Path(self.workdir)

        #####################################
        ### make the setup of the process run


        ### REGION OF EXPERIENCE
        if reg_name == 'north_pacific':
            lats_bnds = np.array([30,65])
            lons_bnds = np.array([120, -120])
        if reg_name == 'north_atlantic':
            lats_bnds = np.array([30,65])
            lons_bnds = np.array([-80, 0])
        if reg_name == 'indian_ocean':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([45, 110])
        if reg_name == 'austral_ocean':
            lats_bnds = np.array([-90,-30])
            lons_bnds = np.array([-180, 180])
        if reg_name == 'tropical_atlantic':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([-70, 20])
        if reg_name == 'tropical_pacific':
            lats_bnds = np.array([-30,30])
            lons_bnds = np.array([120, -70])
        if reg_name == 'mediterranee':
            lats_bnds = np.array([30,50])
            lons_bnds = np.array([-5, 40])
        if reg_name == 'global':
            lats_bnds = np.array([-90,90])
            lons_bnds = np.array([-180,180])

        ### EXPERIENCE NAME
        expname = "ocean_reanalysis_GREP"
        #expname = "sst_retroprevision_sys7"

        ### PERCENTILE THRESHOLD
        percent_thresh = 95
        #percent_thresh = 90

        ### MINIMAL DURATION OF A HW
        duration_min = 5
        #duration_min = 3

        ### YEARS
        if expname == 'ocean_reanalysis_GREP':
            end_year=2016
            start_year=1993
        elif expname == 'sst_retroprevision_sys7':
            end_year=2016
            start_year=1993
        nyear=end_year-start_year+1

        ### SEASON
        #season = "NDJFMAM"
        season = "DJF"
        if season == 'NDJFMAM':
            nday = 211
            season_start_day = [11,1] #1stNov
            season_end_day = [5,31] #31stMay
            first_day = 0
        elif season == 'DJF':
            nday = 90
            season_start_day = [12,1] #1stDec
            season_end_day = [3,1] #28thFeb
            first_day = 30
        ndayseas = nday//duration_min +1

        if expname == "ocean_reanalysis_GREP":
            ### NUMBER OF MEMBS
            first_memb=0
            last_memb=1
            nmemb = last_memb-first_memb

            ### CROSS VALIDATION
            cv = True
            if cv:
                cv_str = "CV"
            else:
                cv_str = 'notCV'

        elif expname == "sst_retroprevision_sys7":
            ### NUMBER OF MEMBS
            first_memb=0
            last_memb=25
            nmemb = last_memb-first_memb    ### NUMBER OF MEMBS

            ### CROSS VALIDATION
            cv = True
            if cv:
                cv_str = "CV"
            else:
                cv_str = 'notCV'

        ##################################
        ### execute the Heatwave Detection
        for (j, jmemb) in enumerate(range(first_memb, last_memb)):
            memb_str = 'memb'+str(jmemb)
            parameters_str = reg_name+"_"+season+"_"+cv_str+'_percent%i'%(percent_thresh)+'_daymin%i'%(duration_min)+"_ref%i-%i_"%(start_year, end_year)+memb_str
            varname = 'subHW_'+parameters_str
            pathHWMI = '/cnrm/pastel/USERS/lecestres/NO_SAVE/data/'+expname+'/'+memb_str+'/'+season+'/'+varname+'/'
            files = glob(pathHWMI + '*.nc')
            files.sort()
            allsubHWslst = []
            def parallel_years(k_start_year, k_end_year):
                year_range = k_end_year-k_start_year
                k_min = k_start_year-start_year
                k_max = k_min+year_range
                for file in files[k_min:k_max]:
                    varf=netCDF4.Dataset(file)
                    varf.variables[varname]
                    vararray, lats_reg, lons_reg = extract_array(varf, varname, ndayseas, np.array(lons_bnds), np.array(lats_bnds),  start_time = 0, level=0)
                    subHWs_iyeararray = vararray[:, np.newaxis, :, :]
                    maskobs = subHWs_iyeararray.mask
                    allsubHWslst.append(subHWs_iyeararray)
                allsubHWs_array = np.ma.array(allsubHWslst)
                #print('maskobs : ', maskobs)
                args = (expname, reg_name, memb_str, season, parameters_str, k_start_year, lats_reg, lons_reg)
                print(allsubHWs_array.shape)
                calc_HW_MY(allsubHWs_array, maskobs, lats_reg, lons_reg, args, allowdist=1)

            CPUs = os.cpu_count()
            years_per_CPU = int(np.ceil(nyear/CPUs))
            k_list = [(start_year + i*(years_per_CPU), min(start_year + (i+1)*years_per_CPU, end_year+1)) for i in range(CPUs)]

            start_time = time.time()
            Parallel(n_jobs=min(CPUs,nyear), backend='multiprocessing', verbose = 20)(delayed(parallel_years)(k_start_year, k_end_year) for (k_start_year, k_end_year) in k_list)
            print('Total time for detection : ', time.time() - start_time)


        ##################################
        ### set the output 

        response.update_status('done.', 100)
        return response
