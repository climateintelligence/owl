# libraries for data analysis

from HW_detection import *

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
        dataset = request.inputs['dataset'][0].file

        response.update_status('Prepare dataset ...', 0)
        workdir = Path(self.workdir)


#TODO: include the analysis code here


        response.update_status('done.', 100)
        return response
