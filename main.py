import pandas as pd
import rpy2

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')
# import R's "utils" package
utils = importr('utils')

brain = pd.read_csv("./data/brain.csv")
main = pd.read_csv("./data/data.csv")
print(brain)

