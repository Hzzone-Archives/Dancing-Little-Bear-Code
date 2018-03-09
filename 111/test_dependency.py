from __future__ import print_function
import urllib.request
import os
from tqdm import tqdm
import shutil
import cntk as C
from cntk import load_model, combine, CloneMethod
from cntk.layers import placeholder
from cntk.logging.graph import find_by_name
import cv2 as cv
import numpy as np
import math
import time
import matplotlib
import pylab as plt
import os
from numpy import ma
from scipy.ndimage.filters import gaussian_filter


# Azure
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path
import sys, io, json, base64, datetime as dt

### 
print("########## All dependencies are installed! ##########")
