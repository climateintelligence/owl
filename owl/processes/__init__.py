from .wps_heatwavedetection import HWs_detection
from .wps_say_hello import SayHello


processes = [
    HWs_detection(),
    SayHello(),
]
