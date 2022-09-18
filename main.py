import pandas as pd
import rpy2

from rpy2.robjects.packages import importr


from helper import *

col = ["family", "sex", "neighbor"]
df = preprosessing(col)
print(df)