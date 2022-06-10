import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import pickle
base = importr('base')
ROSE = importr('ROSE')


def preprosessing(columns, seed=3, size=2000, original_file="aggr-all.csv", orginal=False):
    df = pd.read_csv(f"./data/{original_file}")

    for index, item in enumerate(columns):

        with open(f'./data/columns/{item}', 'rb') as filehandle:
            item = pickle.load(filehandle)
        if(index == 0):
            temp = pd.concat([df[item], df['y']], axis=1)
        else:
            temp = pd.concat([df[item], temp], axis=1)

    if not orginal:
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_from_pd_df = ro.conversion.py2rpy(temp)

        ro.globalenv["df"] = r_from_pd_df
        ro.r(f'''
                if('sex' %in% colnames(df))
                df$sex <- ifelse(df$sex == "M" , 1, 0)
            
                data.rose <- ROSE(y~., data=df, seed={seed},
                        N={size})$data
    ''')

        with localconverter(ro.default_converter + pandas2ri.converter):
            pd_from_r_df = ro.conversion.rpy2py(ro.globalenv["data.rose"])

        assert 'y' in pd_from_r_df
        return pd_from_r_df
    else:
        return temp
