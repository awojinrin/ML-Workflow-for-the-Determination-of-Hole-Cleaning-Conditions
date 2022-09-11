import time
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings('ignore', category = FutureWarning)

def row_optimizer(model, row_data, pct_range, constant_params = [], flex_params = [], all_result = True):
    """
    Helper function for finding optimal parameters within a specified range that minimizes predicted cuttings concentration
    
    Parameters
    -----------
    model : trained model to be used for optimizing drilling parameters
    density : list containing densities to be investigated
    yp : list containing yield points to be investigated
    temp : list containing temperatures to be investigated
    rop: list containing rates of penetration to be investigated
    pipe_rot : list containing pipe rotation speeds to be investigated
    flow_rate : list containing flow rates to be investigated
    inclination : list containing inclinations to be investigated
    eccentricity : list containing eccentricity values to be investigated
    
    Returns
    --------
    This function retuns a dataframe containing the set of parameter values that produces the minimum concentration observed for the given parameter space
    """
    idx_names = {'Density': 'density', 'YP': 'yp', 'Temperature': 'temp', 'ROP': 'rop', 'Pipe rotation': 'pipe_rot', 'Flow rate': 'flow_rate', 
                 'Inclination':'inclination', 'Eccentricity':'eccentricity'}

    row_data.rename(idx_names, inplace = True)
    constant_params = [idx_names[x] for x in constant_params]
    flex_params = [idx_names[x] for x in flex_params]

    myVars = globals()
    for col in set(row_data.index).difference(flex_params).union(constant_params):
        myVars[col] = pd.DataFrame({col: row_data[col]}, index = np.repeat(0, 1))

    for col in set(row_data.index).difference(constant_params).union(flex_params):
        myVars[col] = np.linspace(row_data[col] * (1-pct_range), row_data[col] * (1+pct_range), 10)
        myVars[col] = pd.DataFrame({col: myVars[col]}, index = np.repeat(0, len(myVars[col])))
 
    feature_combinations = pd.merge(density, pd.merge(yp, pd.merge(temp, pd.merge(rop, pd.merge(pipe_rot, 
                            pd.merge(flow_rate, pd.merge(inclination, eccentricity, 'cross'), 'cross'), 
                            'cross'), 'cross'), 'cross'), 'cross'), 'cross')
    feature_combinations['Concentration'] = np.round(model.predict(feature_combinations), 3)

    if not all_result:
        return min(feature_combinations['Concentration'])
    
    return [pd.DataFrame({str(model.__class__).split('.')[-1][:-2]:feature_combinations.iloc[feature_combinations['Concentration'].argmin(),:-1]}), min(feature_combinations['Concentration'])]



def grid_optimizer(model, density, yp, temp, rop, pipe_rot, flow_rate, inclination, eccentricity):
    """
    Helper function for finding optimal parameters within a specified range that minimizes predicted cuttings concentration
    
    Parameters
    -----------
    model : trained model to be used for optimizing drilling parameters
    density : list containing densities to be investigated
    yp : list containing yield points to be investigated
    temp : list containing temperatures to be investigated
    rop: list containing rates of penetration to be investigated
    pipe_rot : list containing pipe rotation speeds to be investigated
    flow_rate : list containing flow rates to be investigated
    inclination : list containing inclinations to be investigated
    eccentricity : list containing eccentricity values to be investigated
    
    Returns
    --------
    This function retuns a dataframe containing the set of parameter values that produces the minimum concentration observed for the given parameter space
    """    

    tic = time.time()
    density = pd.DataFrame({'Density': density}, index = np.repeat(0, len(density)))
    yp = pd.DataFrame({'YP': yp}, index = np.repeat(0, len(yp)))
    temp = pd.DataFrame({'Temperature': temp}, index = np.repeat(0, len(temp)))
    rop = pd.DataFrame({'ROP': rop}, index = np.repeat(0, len(rop)))
    pipe_rot = pd.DataFrame({'Pipe rotation': pipe_rot}, index = np.repeat(0, len(pipe_rot)))
    flow_rate = pd.DataFrame({'Flow rate': flow_rate}, index = np.repeat(0, len(flow_rate)))
    inclination = pd.DataFrame({'Inclination': inclination}, index = np.repeat(0, len(inclination)))
    eccentricity = pd.DataFrame({'Eccentricity': eccentricity}, index = np.repeat(0, len(eccentricity)))
    
    feature_combinations = pd.merge(density, pd.merge(yp, pd.merge(temp, pd.merge(rop, pd.merge(pipe_rot, 
                            pd.merge(flow_rate, pd.merge(inclination, eccentricity, 'cross'), 'cross'), 
                            'cross'), 'cross'), 'cross'), 'cross'), 'cross')
    feature_combinations['Concentration'] = np.round(model.predict(feature_combinations), 3)
    
    toc = time.time()
    
    print(str(model.__class__).split('.')[-1][:-2])
    print('-----------------------------------------------------------------------')
    print(f'Time taken to search parameter space containing {len(feature_combinations)} combinations: {round(toc - tic, 3)}s')
    print('-----------------------------------------------------------------------')
    
    return pd.DataFrame({str(model.__class__).split('.')[-1][:-2]:feature_combinations.iloc[feature_combinations['Concentration'].argmin()]})