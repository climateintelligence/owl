from .wps_say_hello import SayHello
from .wps_heatwavedetection import HWs_detection


processes = [
    SayHello(),
    HWs_detection(),
]
