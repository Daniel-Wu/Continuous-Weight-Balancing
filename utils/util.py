import numpy as np
from bokeh.plotting import figure
from sklearn.metrics import mean_squared_error, r2_score


def plot_cont_var(column_name, df):
    """
    A function which makes histogram for continuous variables.
    """
    hist1, edges1 = np.histogram(df[df["target"] == 0][column_name], density = True, bins = 40)
    hist2, edges2 = np.histogram(df[df["target"] == 1][column_name], density = True, bins = 40)

    p = figure(
        plot_height = 500,
        plot_width = 500,
        x_axis_label = column_name,
        title = column_name.capitalize() + ' vs Target'
    )

    p.quad(
        bottom = 0,
        top = hist1,
        left = edges1[:-1],
        right = edges1[1:],
        line_color = 'white',
        color = 'blue', # Blue represents patients not having heart disease.
        alpha = 0.6
    )

    p.quad(
        bottom = 0,
        top = hist2,
        left = edges2[:-1],
        right = edges2[1:],
        line_color = 'white',
        color = 'red', # Red represents patients having heart disease.
        alpha = 0.6
    )
    return p

def roc_sklearn_model(model, X, y):
    """
    Validates a shallow sklearn model on X & y and returns ROC value
    """
    rf_probs = model.predict_proba(X)[:, 1]
    roc_value = roc_auc_score(y, rf_probs)
    return roc_value   

def assess_sklearn_regressor(model, X, y):
    """
    Validates a shallow sklearn model on X & y and returns ROC value
    """
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    return r2, mse

def plot_loss(history):
    plt.title('Loss curves')
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()
