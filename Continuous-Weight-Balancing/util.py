import numpy as np
from bokeh.plotting import figure


def plot_cont_var(column_name):
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

