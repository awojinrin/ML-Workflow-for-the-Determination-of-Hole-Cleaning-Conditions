import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV


def data_split(df, test_size = 0.25, random_state = 42):
    features = df.drop(columns = 'Concentration')
    target = df.Concentration

    return train_test_split(features, target, test_size = .25, random_state = 42)


def rsearch(iter:int, est, params, x, y, n_cv = 3):
    randomized_search = RandomizedSearchCV(estimator=est, param_distributions=params, n_iter=iter, scoring='neg_root_mean_squared_error', verbose=0, cv = n_cv, n_jobs=-1)
    randomized_search.fit(x, y)
    return randomized_search


def gsearch(est, params, x, y, n_cv = 3):
    grid_search = GridSearchCV(estimator= est, param_grid=params, scoring='neg_root_mean_squared_error', verbose=0, cv = n_cv, n_jobs=-1)
    grid_search.fit(x, y)
    return grid_search


def model_metric(model, trainX, trainy, testX, testy, silent: bool = False):
    """
    Helper function that trains the model and outputs the MAE and R2 Score for a given model
    
    Parameters:
    -- model: the model to be trained and evaluated
    -- trainX: input features for the training stage
    -- trainy: target for the training stage
    -- testX: input features for the testing stage
    -- testy: target for the testing stage
    
    Returns:
    training_predictions, test_predictions
    
    """
    
    model.fit(trainX, trainy)
    train_prediction = model.predict(trainX)
    test_prediction = model.predict(testX)
        
    train_mae = mae(trainy, train_prediction)
    test_mae = mae(testy, test_prediction)

    train_r2_score = r2(trainy, train_prediction)
    test_r2_score = r2(testy, test_prediction)
    
    if not silent:
        print('TRAINING DATASET')
        print('MAE: ', round(train_mae*100, 2), '%')
        print('R2_SCORE: ', train_r2_score)
        print('----------------------------')    
        print('TEST DATASET')
        print('MAE: ', round(test_mae*100, 2), '%')
        print('R2_SCORE: ', test_r2_score)
        print('----------------------------')    
    
    return train_prediction, test_prediction


def model_visuals(y_train, y_test, train_pred, test_pred, filename = str(None), model_type = str(None)):
    """
    Helper function for easy visualization of model performance

    Parameters:
    -- y_train: target for the training stage
    -- y_test: target for the testing stage
    -- train_pred: predictions for the training stage
    -- test_pred: predictions for the testing stage
    -- filename: str, name of output file to save image to; default None
    -- model_type: str, the model to be visualized; default None

    """

    plt.rc('axes', labelsize = 17)
    plt.rc('font', size = 17, weight = 'bold')
    sns.set_style('darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (14, 6))
    fig.suptitle(model_type)
    
    min_val = np.min([y_train.min(), y_test.values.min(), train_pred.min(), test_pred.min()]) - 0.01
    max_val =  np.max([y_train.max(), y_test.values.max(), train_pred.max(), test_pred.max()]) + 0.01
    
    ax1.plot([min_val, max_val], [min_val, max_val], '--k', linewidth = 2, label = 'Perfect Prediction')
    ax1.scatter(y_train, train_pred, s = 20, c = 'blue')
    ax1.legend(loc = 'lower right')
    ax1.text(min_val+0.02, max_val-0.05, 'R2: ' + str(round(r2(y_train, train_pred), 3)) + '\nMAE: ' + str(round(mae(y_train, train_pred)*100, 2)) + '%')
    ax1.set_ylabel('Predicted Cutting Concentration')
    ax1.set_xlabel('Observed Cutting Concentration')
    ax1.set_title('Training Set', {'fontsize': 17, 'fontweight': 'bold'}, 'left')
    
    ax2.plot([min_val, max_val], [min_val, max_val], '--k', linewidth = 2, label = 'Perfect Prediction')
    ax2.scatter(y_test, test_pred, s = 20, c = 'blue')
    ax2.legend(loc = 'lower right')
    ax2.text(min_val+0.02, max_val-0.05, 'R2: ' + str(round(r2(y_test, test_pred), 3)) + '\nMAE: ' + str(round(mae(y_test, test_pred)*100,2) ) + '%')
    ax2.set_ylabel('Predicted Cutting Concentration')
    ax2.set_xlabel('Observed Cutting Concentration')
    ax2.set_title('Test Set', {'fontsize': 17, 'fontweight': 'bold'}, 'right')
    
    fig.tight_layout()
    plt.show()
    
    if filename != None:
        plt.savefig(fname = 'files/' + str(filename) + '.png', format = 'png')


def model_report(model, trainX, trainy, testX, testy, filename :str = None, model_type: str = None):
    """
    Helper function for training the model, as well as easy evaluation and visualization of model performance.
    A combinaton of model_metric and model_visuals
    
    Parameters:
    -- model: the model to be evaluated
    -- trainX: input features for the training stage
    -- trainy: target for the training stage
    -- testX: input features for the testing stage
    -- testy: target for the testing stage
    -- filename: str, name of output file to save image to; default None
    -- model_type: str, the model to be visualized; default None
    """
    
    train_pred, test_pred = model_metric(model, trainX, trainy, testX, testy)
    model_visuals(trainy, testy, train_pred, test_pred, filename, model_type)
