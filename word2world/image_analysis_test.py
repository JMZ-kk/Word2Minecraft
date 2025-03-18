import base64
import requests
import os
import openai
from configs import Config
from w2m_tools import *

print(get_common_advice("word2world/images/example_1_tpd.png",
                        "The following image is the top-down view of a game level.Can you descript this image?"))
# print(img_analysis("word2world/images/example_1_tpd.png"))