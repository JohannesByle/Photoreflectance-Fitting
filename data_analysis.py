import os
import re
import datetime
import scipy
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from matplotlib.widgets import Slider
from matplotlib import cm


# take data set and return the r-squared value
def RSQ(data, model):

    # Calculate squared error
    def squared_error(data, model):
        return sum((model - data) * (model - data))

    y_mean_line = [np.mean(data) for y in data]
    squared_error_regr = squared_error(data, model)
    squared_error_y_mean = squared_error(data, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


# return the value closest to given value for given array (from unutbu on stack overflow)
def find_nearest(array, value):
    array = np.asarray(array)
    value = float(value)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# cut x and y data to be within a certain range in x
def cut(x_data, y_data, cut_range):

    # Find the indexes to cut
    min_index = x_data.index(find_nearest(x_data, cut_range[0]))
    max_index = x_data.index(find_nearest(x_data, cut_range[1]))

    return x_data[min_index:max_index], y_data[min_index:max_index]


# takes directory and file identifier and returns the numeric values in the file as arrays
def import_file(directory, pattern, identifier="", return_file=False):
    """
    Takes directory and file identifier and returns the numeric values in the file either as arrays or a single values

    Parameters
    ----------
    directory : str
        directory in which data files are found or file path
    pattern : regex
        regex expression which signifies what part of the filename should be saved
    identifier : str
        the starting variables of the filenames, if none is given all .txt files are opened
    return_file : bool
        if 'True' pattern is ignored and filename is returned as is

    Returns
    -------
    filename_array : list of list of str
        array of either filenames or values extracted from filename, if only file is passed as an argument only a str
        is returned
    data_array : list of list of list of float
        multidimensional array of values extracted from file. [file][line in file][value in row], if only file is passed
        as an argument list of list of str is returned
    """

    # Sub function to open a single file rather than directory
    def get_file(filename, pattern, identifier="", return_file=False):

        # Open file
        file = open(filename, mode='r')

        # Array to hold extracted values
        data = []

        # Loop over lines in file and extract numbers avoiding errors from titles
        # noinspection PyBroadException
        try:
            for line in file:

                # Use regex to extract all numbers from a line
                line = re.findall(re.compile(r'-?\d+\.?\d*E?-?\d*'), line)
                line = [float(n) for n in line]

                # Check if line did not contain any numbers
                if line:
                    data.append(line)
                else:
                    continue

        except Exception:
            pass

        # Extract data from filename
        if not return_file:
            filename = [float(n) for n in pattern.findall(filename)]

        return filename, data

    # Define arrays to hold the data and filename
    data_array = []
    filename_array = []

    if os.path.isdir(directory):
        # Loop over the files in the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):

                    # Convert filename to full path
                    filename = os.path.join(root, file)

                    # Extract data from file
                    # Filter out files based on identifier
                    if os.path.basename(filename).startswith(identifier) or identifier == "":
                        filename, data = get_file(filename, pattern, identifier, return_file)
                    else:
                        continue

                    # Store data and data from filename
                    data_array.append(data)
                    filename_array.append(filename)

        return zip(*sorted(zip(filename_array, data_array)))

    else:
        return get_file(directory, pattern, identifier, return_file)


# takes a list of x values and y values and fits them all to a function
def fit_all(x_data, y_data, guess_data, fit_func):
    """
    Takes a list of x values and y values and fits them all to a function

    Parameters
    ----------
    x_data : list of list of float
        set of x data where rows of array correspond to sets of x data
    y_data : list of list of float
        set of y data where rows of array correspond to sets of y data
    guess_data : list of list of float
        set of guess values that has the same number of rows as the x and y data and the same number of columns as
        the number of fitting coefficients as the function
    fit_func : func
        function to which the data is fit, must be in the form that curve_fit accepts
    Returns
    -------
    fit_array : list of float
        array of fit values which has the same shape as guess_data
    rsq_array : list of float
        array of r_squared values
    """

    # Array to hold fit and RSQ values
    fit_array = []
    rsq_array = []

    # Loop over x and y data and fit each one, if it doesn't work plot data and guess
    for n in range(0, len(x_data)):

        # noinspection PyBroadException
        try:
            fit, _ = curve_fit(fit_func, x_data[n], y_data[n], guess_data[n])
            fit_array.append(fit)
            rsq_array.append(RSQ(np.asarray(y_data[n]), fit_func(np.asarray(x_data[n]), *guess_data[n])))
        except Exception as e:
            plt.scatter(x_data[n], y_data[n])
            plt.plot(x_data[n], fit_func(np.asarray(x_data[n]), *guess_data[n]), 'r')
            plt.title("There was an error in the fit")
            plt.show()
            print(e)

    return fit_array, rsq_array


# takes a list of x and y values and returns the fwhm
def fwhm_all(x_data, y_data, data_range):
    """
    Takes a list of x values and y values and returns the fwhm of data. Assumes there is a gaussian peak in the range

    Parameters
    ----------
    x_data : list of list of float
        set of x data where rows of array correspond to sets of x data
    y_data : list of list of float
        set of y data where rows of array correspond to sets of y data
    data_range : list of float
        range in which the data is found, should be a array of two values where the first value corresponds to the
        minimum value in x and the second value is the largest value in x

    Returns
    -------
    fwhm_array : list of float
        an array where each row contains the fwhm of the given x and y data
    """

    # Array to hold guess values
    fwhm_array = []

    # Loop over x and y data and guess values for each one
    for n in range(0, len(x_data)):

        # Convert range to index
        min_index = x_data[n].index(find_nearest(x_data[n], data_range[0]))
        max_index = x_data[n].index(find_nearest(x_data[n], data_range[1]))

        # Cut x and y to range
        new_x_data = x_data[n][min_index:max_index]
        new_y_data = y_data[n][min_index:max_index]

        # Create new max index based on peak
        max_index = new_y_data.index(max(new_y_data))

        # Calculate fwhm
        integral_max = np.trapz(new_y_data[:max_index], new_x_data[:max_index])
        interp_function = np.interp(np.linspace(min(new_x_data), max(new_x_data), 1000), new_x_data, new_y_data)

        # Append fwhm to array
        fwhm_array.append(2*abs(find_nearest(interp_function, integral_max/2)-new_x_data[max_index]))

    return fwhm_array


# Function that takes x values and coefficients and returns the value predicted by the one Gaussian function
def one_gaussian(x, a, b, c):
    r'''
    :math:`\frac{A}{c\sqrt{\frac{\pi}{4\ln(2)}}}\exp(\frac{-4\ln(2)(x-b)^2}{c^2})`
    '''
    if a < 0 or b < 0 or c < 0:
        return np.inf
    return a*(1/(c*np.sqrt(np.pi/(4*np.log(2)))))*np.exp((-4*np.log(2)*(x-b)**2)/(c**2))


# guesses fit parameters for a gaussian fit
def gaussian_guess(x_data, y_data, data_range):
    """
    Calculate a guess for starting parameters for the fit

    Parameters
    ----------
    x_data : list of list of float
        list of x data
    y_data : list of list of float
        list of y data
    data_range : list of float
        x range of data, [min, max]
    Returns
    -------
    guess_array : list of list of float
        array of guess arrays, each row in array is a list of values for the fit function variables
    """
    # Array to hold guess values
    guess_array = []

    # Loop over x and y data and guess values for each one
    for n in range(0, len(x_data)):

        # Convert range to index
        min_index = x_data[n].index(find_nearest(x_data[n], data_range[0]))
        max_index = x_data[n].index(find_nearest(x_data[n], data_range[1]))

        # Cut x and y to range
        new_x_data = list(x_data[n][min_index:max_index])
        new_y_data = list(y_data[n][min_index:max_index])

        # Make guesses for the variables
        c = (max(new_x_data)-min(new_x_data))/4
        a = max(new_y_data)/(1/(c*np.sqrt(np.pi/(4*np.log(2)))))
        b = new_x_data[new_y_data.index(max(new_y_data))]
        # Add variables to the array
        guess_array.append([a, b, c])

    return guess_array


# Function that takes x values and coefficients and returns the value predicted by the one Gaussian function
def one_gaussian_with_d(x, a, b, c, d):
    r'''
    :math:`d+\frac{A}{c\sqrt{\frac{\pi}{4\ln(2)}}}\exp(\frac{-4\ln(2)(x-b)^2}{c^2})`
    '''
    if a < 0 or b < 0 or c < 0:
        return np.inf
    return a*(1/(c*np.sqrt(np.pi/(4*np.log(2)))))*np.exp((-4*np.log(2)*(x-b)**2)/(c**2))+d


# guesses fit parameters for a gaussian fit
def gaussian_guess_with_d(x_data, y_data, data_range):
    """
    Calculate a guess for starting parameters for the fit

    Parameters
    ----------
    x_data : list of list of float
        list of x data
    y_data : list of list of float
        list of y data
    data_range : list of float
        x range of data, [min, max]
    Returns
    -------
    guess_array : list of list of float
        array of guess arrays, each row in array is a list of values for the fit function variables
    """
    # Array to hold guess values
    guess_array = []

    # Loop over x and y data and guess values for each one
    for n in range(0, len(x_data)):

        # Convert range to index
        min_index = x_data[n].index(find_nearest(x_data[n], data_range[0]))
        max_index = x_data[n].index(find_nearest(x_data[n], data_range[1]))

        # Cut x and y to range
        new_x_data = list(x_data[n][min_index:max_index])
        new_y_data = list(y_data[n][min_index:max_index])

        # Make guesses for the variables
        c = (max(new_x_data)-min(new_x_data))/4
        a = max(new_y_data)/(1/(c*np.sqrt(np.pi/(4*np.log(2)))))
        b = new_x_data[new_y_data.index(max(new_y_data))]

        # Add variables to the array
        guess_array.append([a, b, c, min(new_y_data)])

    return guess_array


# Function that takes x values and coefficients and returns the value predicted by the Aspnes equation
def aspnes_equation(Energy, amplitude, phase, Eg, gamma):
    r'''
    :math:`\Re\big[Ae^{i\gamma}(E-E_g+i\Gamma)^{-3}\big]`
    '''
    i = complex(0, 1)
    return np.real(amplitude*np.exp(i*phase)*(Energy-Eg+i*gamma)**(-3))


# guesses fit parameters for a Aspnes fit
def aspnes_guess(x_data, y_data, data_range):
    """
    Calculate a guess for starting parameters for the fit

    Parameters
    ----------
    x_data : list of list of float
        list of x data
    y_data : list of list of float
        list of y data
    data_range : list of float
        x range over which the data is fit, [min, max]
    Returns
    -------
    guess_array : list of list of float
        array of guess arrays, each row in array is a list of values for the fit function variables
    """

    # Array to hold guess values
    guess_array = []

    # Loop over x and y data and guess values for each one
    for n in range(0, len(x_data)):

        # Make sure x and y data are oriented correctly
        x_data[n], y_data[n] = zip(*sorted(zip(x_data[n], y_data[n])))

        # Convert range to index
        min_index = x_data[n].index(find_nearest(x_data[n], data_range[0]))
        max_index = x_data[n].index(find_nearest(x_data[n], data_range[1]))

        # Cut x and y to range
        new_x_data = x_data[n][min_index:max_index]
        new_y_data = y_data[n][min_index:max_index]

        # Make guesses for the variables
        r_min = new_x_data[np.argmin(new_y_data)]
        r_max = new_x_data[np.argmax(new_y_data)]
        gamma = np.abs(r_max - r_min) * 0.5 * (np.sqrt(2) + 1)
        Eg = np.abs(r_max + r_min) * 0.5
        r_max = np.max(np.abs(new_y_data))
        c_max = gamma ** 3 * r_max
        c_min = -(32 * gamma ** 6 * r_max) / ((10 + 7 * np.sqrt(2)) * (2 * (np.sqrt(2) - 2) * gamma ** 3))
        amplitude = (c_min + c_max) * 0.5
        phase = 0

        # Add variables to the array
        guess_array.append([amplitude, phase, Eg, gamma])

    return guess_array


# Function that takes x values and coefficients and returns the value predicted by the broadening equation
def broadening_fit(T, g0, g_la, g_lo, o_lo, g_imp):
    r'''
    :math:`\Gamma_0+\Gamma_{LA}T+\frac{\Gamma_{LO}}{\big[\exp(\frac{h\omega_{LO}}{k_BT})\big]}+
    \Gamma_{Imp}\exp(-\frac{E_B}{k_BT_H})`
    '''
    if g0 <0 or g_la < 0 or  g_lo < 0 or o_lo < 0 or g_imp < 0:
        return np.inf
    k_b = 8.6173 * 10 ** (-2)
    return g0+g_la*T+g_lo/(np.exp(o_lo/(k_b*T))-1)+g_imp*np.exp((-5) / (k_b * T))


# create ranges of min and max for a given array and percentile
def plot_scale(array, percent):
    """
    Create ranges of min and max for a given array and percentile

    Parameters
    ----------
    array : list
        array to be plotted
    percent : float
        percent of array to be included as padding on either side of the array

    Returns
    -------
    new_range : list
        new min and max scale values
    """

    array_range = max(array) - min(array)
    return [min(array) - array_range * percent, max(array) + array_range * percent]


# creates waterfall plot for a given set of x and y data and optionally includes the fit
def waterfall(x_data, y_data, s=.00001, size=0.25, log=True, offset=1000, title="Waterfall Plot of Data",
              xlabel="Wavelength (nm)", ylabel="Signal (arb. units)", fit_func=0, guess_data=0, scatter=True):
    """
    Creates waterfall plot for a given set of x and y data and optionally includes the fit

    Parameters
    ----------
    x_data : list of list of float
        set of x data where rows of array correspond to sets of x data
    y_data : list of list of float
        set of y data where rows of array correspond to sets of y data
    s : float
        percent increase between each plot
    size : float
        size of points or line on graph
    log : bool
        variable determining whether graph should be scaled logarithmically
    offset : float
        amount each plot should be offset from the x axis
    title : str
    xlabel : str
    ylabel : str
    fit_func : func
        function to which the data should be fit, must be in format that curve_fit accepts
    guess_data : list of list of float
        set of guess values that has the same number of rows as the x and y data and the same number of columns as the
        number of fitting coefficients as the function
    scatter : bool
        variable determining whether data should be plotted as a scatter plot or line plot

    Returns
    -------

    """

    # Create step value for waterfall
    s = (max([max(n) for n in y_data])-min([min(n) for n in y_data]))*s

    # Loop over rows in x_data and plot them
    for n in range(0, len(x_data)):

        # Convert x and y data to np for plotting
        x_data[n] = np.asarray(x_data[n])
        y_data[n] = np.asarray(y_data[n])

        if scatter:
            plt.scatter(x_data[n], y_data[n]+n*s+offset, s=size)
        else:
            plt.plot(x_data[n], y_data[n] + n * s + offset, linewidth=size)
        if fit_func != 0:
            plt.plot(x_data[n], fit_func(x_data[n], *guess_data[n])+n*s)

    # Create log plot based on func args
    if log:
        plt.yscale("log")

    # Change plot settings
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    plt.tick_params(labelsize=20)

    plt.show()


# plot a given set of variables and the fit
def fit_plot(x_data, y_data, fit_func, fit_values, fit_variables, x_label="X Axis", y_label="Y Axis", title="Title"):
    """
    Plot a given set of variables and the fit

    Parameters
    ----------
    x_data : list of float
        set of x_data where rows of array correspond to sets of x_data
    y_data : list of float
        set of y_data where rows of array correspond to sets of y_data
    fit_func : func
        function to which the data is fit, must be in the form that curve_fit accepts
    fit_values : list of float
        values to be passed to the fit_func
    fit_variables : list of str
        string array which is used for the legend of the plot
    x_label : str
    y_label : str
    title : str

    Returns
    -------

    """

    # Plot the data points
    plt.scatter(x_data, y_data, s=50)

    # Create the label
    label = "Fit"
    for n in range(0, len(fit_values)):
        label = label + ", " + str(fit_variables[n]) + " = " + str(round(fit_values[n], 2))

    # Create the range and plot the fit
    x_range = np.linspace(min(x_data), max(x_data), 1000)
    plt.plot(x_range, fit_func(x_range, *fit_values), label=label, color='red', linewidth=2.5)

    # Edit the plot variables to make them better for presenting
    plt.title(title, fontsize=24)
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    plt.tick_params(labelsize=20)
    plt.legend(prop={'size': 20})

    plt.show()


# create a slider to check fits of files in a directory
def fit_plot_all(x_data, y_data, fit_func, fit_values, fit_variables, slider_range):
    """
    Create a slider to check fits of files in a directory

    Parameters
    ----------
    x_data : list of list of float
        set of x_data where rows of array correspond to sets of x_data
    y_data : list of list of float
        set of y_data where rows of array correspond to sets of y_data
    fit_func : func
        function to which the data is fit, must be in the form that curve_fit accepts
    fit_values : list of list of float
        values to be passed to the fit_func
    fit_variables : list of str
        string array which is used for the legend of the plot
    slider_range : list of float
        range over which the slider should go, should be an array of the same length as x and y data

    Returns
    -------

    """

    # Create figure
    fig = plt.figure()

    # Define shape of axes
    plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    # Initialize the slider
    slider = Slider(slider_ax, 'Offset', slider_range[0], slider_range[len(slider_range)-1], slider_range[0], valstep=1)

    # Updates the slider
    def update(val):
        # noinspection PyBroadException
        try:
            # Convert from value to index
            n = slider_range.index(val)

            # Name axes to be plotted and clear them
            plt.axes(plot_ax)
            plt.cla()

            # Plot the data points
            plt.scatter(x_data[n], y_data[n])
            plt.ylim(plot_scale(y_data[n], 0.25))

            # Create the label
            label = "Fit"
            for i in range(0, len(fit_values[n])):
                label = label + ", " + str(fit_variables[i]) + " = " + str(round(fit_values[n][i], 2))

            # Create the range and plot the fit
            x_range = np.linspace(min(x_data[n]), max(x_data[n]), 1000)
            plt.plot(x_range, fit_func(x_range, *fit_values[n]), label=label, color='red')
            plt.title(slider_range[n])
            plt.legend()
            fig.canvas.draw_idle()
        except Exception:
            pass

    # Run once to not show blank graph on startup
    update(slider_range[0])

    # Show the plot
    slider.on_changed(update)
    plt.show()


# create a slider to check files in a directory
def plot_all(x_data, y_data, slider_range, x_label="X Axis", y_label="Y Axis", title="Title"):
    """
    Create a slider to check fits of files in a directory

    Parameters
    ----------
    x_data : list of list of float
        set of x_data where rows of array correspond to sets of x_data
    y_data : list of list of float
        set of y_data where rows of array correspond to sets of y_data
    slider_range : list of float
        range over which the slider should go, should be an array of the same length as x and y data
    x_label : str
    y_label : str
    title : str

    Returns
    -------

    """

    # Create figure
    fig = plt.figure()

    # Define shape of axes
    plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
    slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

    # Initialize the slider
    slider = Slider(slider_ax, 'Offset', slider_range[0], slider_range[len(slider_range)-1], slider_range[0], valstep=1)

    # Updates the slider
    def update(val):
        # noinspection PyBroadException
        # Convert from value to index
        n = slider_range.index(val)

        # Name axes to be plotted and clear them
        plt.axes(plot_ax)
        plt.cla()
        # Plot the data points
        plt.scatter(x_data[n], y_data[n])
        if title == "Title":
            plt.title(slider_range[n], fontsize=24)
        else:
            plt.title(title, fontsize=24)
        plt.xlabel(x_label, fontsize=22)
        plt.ylabel(y_label, fontsize=22)
        plt.tick_params(labelsize=20)
        fig.canvas.draw_idle()

    # Run once to not show blank graph on startup
    update(slider_range[0])

    # Show the plot
    slider.on_changed(update)
    plt.show()


# create plots of integrated intensity and peak intensity vs temp or bias for files in a directory
def integrated_i_plot(start_directory, wavelength_range, jv_identifier="NaN", pl_identifier=".txt", td=True,
                      plot_type='sub', time=False, temps="all"):
    """
    Create plots of integrated intensity and peak intensity vs temp or bias for files in a directory

    Parameters
    ----------
    start_directory : str
        directory in which files are found
    wavelength_range : list of float
        wavelength range over which should be integrated and peak should be found, [min, max]
    jv_identifier : str
        starting string of jv files
    pl_identifier : str
        starting string of pl files
    td : bool
        whether or not the subplots should be organized by temperature or bias
    plot_type : str
        'sub' or 'single', determines how data should be plotted
    time : bool
        if true data is plotted against file creation time
    temps : list of float
        temperatures to be included in the plot

    Returns
    -------

    """

    # Initialize the arrays to store the values
    integrated_intensity = []
    peaks = []
    bias_data = []
    wavelength_data = []
    signal_data = []
    temp_data = []
    jv_temp = []
    voltage = []
    current = []
    time_data = []

    # Loop over the files in the directory
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if file.endswith(".txt"):

                # Try in order to avoid errors from files without bias or temp data
                try:
                    # Convert filename to  full path
                    directory = os.path.join(root, file)

                    # Import different information if files are JV files
                    if jv_identifier != "NaN" and file.startswith(jv_identifier):
                        filename, data = import_file(directory, re.compile(r'_(\d+)K'), jv_identifier)
                        jv_temp.append(filename[0])
                        voltage.append([i[0] for i in data])
                        current.append([i[1] for i in data])
                        continue

                    if file.startswith(pl_identifier):
                        filename, data = import_file(directory, re.compile(r'_(-?\d+.?\d*)[V,K]'), pl_identifier)
                        # If time data is wanted import time data
                        if time:
                            time_file, _ = import_file(directory, re.compile(r'_(-?\d+.?\d*)[V,K]'), pl_identifier, time)
                            time_data.append(datetime.datetime.fromtimestamp(os.path.getmtime(time_file)))
                    else:
                        continue
                    # Convert add data to arrays
                    wavelength_data.append([i[0] for i in data])
                    signal_data.append([i[1] for i in data])

                    # Determine if it should be bias, temperature or time dependent
                    if td:
                        # Add bias and temp data to array
                        bias_data.append(filename[0])
                        temp_data.append(filename[1])

                    else:
                        # Add bias and temp data to array
                        bias_data.append(filename[1])
                        temp_data.append(filename[0])

                    # If plotting by time make x_data time
                    if time:
                        bias_data = time_data.copy()

                    # Add peak and integrated data to array
                    j = len(signal_data)-1
                    e_min = list(wavelength_data[j]).index(find_nearest(wavelength_data[j], wavelength_range[0]))
                    e_max = list(wavelength_data[j]).index(find_nearest(wavelength_data[j], wavelength_range[1]))
                    integrated_intensity.append(integrate.simps(signal_data[j][e_min:e_max], wavelength_data[j][e_min:e_max]))
                    peaks.append(max(signal_data[j]))
                except IndexError:
                    pass

    # Initialize the shaped arrays
    temp_array = []
    integrated_array = []
    peaks_array = []
    bias_array = []

    # Loop over temp array and for each temp create a row in other arrays
    for n in range(len(temp_data)):
        if temps == 'all' or temp_data[n] in temps:
            # If temp is already in array append to row
            if temp_data[n] in temp_array:
                i = temp_array.index(temp_data[n])
                integrated_array[i].append(integrated_intensity[n])
                peaks_array[i].append(peaks[n])
                bias_array[i].append(bias_data[n])

            # If temp is new append new row with array of each value
            else:
                temp_array.append(temp_data[n])
                integrated_array.append([integrated_intensity[n]])
                peaks_array.append([peaks[n]])
                bias_array.append([bias_data[n]])

    # Sort arrays
    temp_array, integrated_array, peaks_array, bias_array = \
        zip(*sorted(zip(temp_array, integrated_array, peaks_array, bias_array)))

    # Function to plot data in subplots
    def subplot_plot(y_data, title, x_label, y_label, legend, color):

        # Create a plot of integrated intensity vs bias
        fig, axes = plt.subplots(nrows=2, ncols=int((len(temp_array) + 1.5)/2), figsize=(16, 9))

        # Create title, x and y labels
        if time:
            fig.text(0.5, 0.03, "Time", ha='center', va='center', fontsize=22)
        else:
            fig.text(0.5, 0.03, x_label, ha='center', va='center', fontsize=22)
        fig.text(0.03, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=22,
                 color=color)
        if jv_identifier != "NaN":
            fig.text(0.95, 0.5, "Current (mA)", ha='center', va='center', rotation='vertical', fontsize=22)

        # Loop over row in temp array and plot each
        for ax, n in zip(axes.flatten(), range(len(temp_array))):
            a = 1
            b = 0.15

            plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
            # Create each subplot
            plt.axes(ax)

            # Extra line to avoid errors when plotting time
            plt.plot([], [])

            # Plot the data
            plt.scatter(bias_array[n], y_data[n], color=color, label=str(temp_array[n])+legend)

            # Create range so that data is properly scaled
            plt.ylim(plot_scale(y_data[n], a))
            plt.xlim(plot_scale(bias_array[n], b))

            # Create titles and other writing
            plt.tick_params(labelsize=18)
            plt.suptitle(title+str(min(temp_array)) + legend + " to " + str(max(temp_array)) + legend, color='black',
                         fontsize=24)
            plt.legend(fontsize=14)

            if time:
                plt.gcf().autofmt_xdate()
                myFmt = mdates.DateFormatter('%H:%M')
                plt.gca().xaxis.set_major_formatter(myFmt)

            # Plot the JV if JV is found
            if jv_identifier != "NaN":
                i = jv_temp.index(temp_array[n])
                ax2 = plt.twinx()
                plt.axes(ax2)
                plt.plot(voltage[i], current[i], color='black')
                plt.tick_params(labelsize=18)

        # Save the figure
        plt.savefig(start_directory + "\\" + title + str(min(temp_array)) + legend + " to " + str(max(temp_array)) + legend +
                    ".svg")

        plt.show()

    def regular_plot(y_data, title, x_label, y_label, legend):

        # Create figure and titles
        plt.figure(figsize=(16, 9))
        plt.tight_layout()
        plt.title(title+str(min(temp_array)) + legend + " to " + str(max(temp_array)) + legend, color='black',
                  fontsize=24)
        if time:
            plt.xlabel("time", fontsize=22)
        else:
            plt.xlabel(x_label, fontsize=22)
        plt.ylabel(y_label, fontsize=22)

        # Plot all data on same plot
        for n in range(len(temp_array)):
            plt.scatter(bias_array[n], y_data[n], label=str(temp_array[n])+legend)

        plt.legend(fontsize=14)
        plt.tick_params(labelsize=18)

        # Save the figure
        plt.savefig(
            start_directory + "\\" + title + str(min(temp_array)) + legend + " to " + str(max(temp_array)) + legend +
            ".svg")

        plt.show()

    # Plot the data on either subplots or a single plot
    if plot_type == 'sub':
        # Plot the data with different titles depending on how it is plotted
        if td:
            subplot_plot(integrated_array, "Integrated Intensity vs Bias at ", "Bias (V)", "Integrated Intensity (V)", "K",
                         'blue')
            subplot_plot(peaks_array, "Peak Intensity vs Bias at ", "Bias (V)", "Peak Intensity (V)", "K", 'red')
        else:
            subplot_plot(integrated_array, "Integrated Intensity vs Temperature at ", "Temperature (K)",
                         "Integrated Intensity (V)", "V", 'blue')
            subplot_plot(peaks_array, "Peak Intensity vs Temperature at ", "Temperature (K)", "Peak Intensity (V)", "V",
                         'red')
    elif plot_type == 'single':
        # Plot the data with different titles depending on how it is plotted
        if td:
            regular_plot(integrated_array, "Integrated Intensity vs Bias at ", "Bias (V)", "Integrated Intensity (V)",
                         "K")
            regular_plot(peaks_array, "Peak Intensity vs Bias at ", "Bias (V)", "Peak Intensity (V)", "K")
        else:
            regular_plot(integrated_array, "Integrated Intensity vs Temperature at ", "Temperature (K)",
                         "Integrated Intensity (V)", "V")
            regular_plot(peaks_array, "Peak Intensity vs Temperature at ", "Temperature (K)", "Peak Intensity (V)", "V")


# Plot and/or fit data in files in a given directory
def plot_directory(directory, pattern, x_range, x_col=0, y_col=1, fit_func="none", guess_func="none",
                   plot_type='waterfall', scatter_size=1, linewidth=1, color='range', offset="auto", rmv_baseline=False,
                   subplot_values="all", return_plot=False, fit_variable="none", identifier=""):
    """
    Plot and/or fit data in files in a given directory

    Parameters
    ----------
    directory : str
        path of directory containing files to be plotted
    pattern : regex
        pattern to be extracted from the filename
    x_range : list of float
        x range over which the data is fit, [min, max]
    x_col : int
        column in txt file which contains x data
    y_col : int
        column in txt file which contains y data
    fit_func : func
         function to which the data is fit, must be in the form that curve_fit accepts
    guess_func :func
        function which generates the guess data must have form f(x_data, y_data, range)
    plot_type : str
        type of plot: waterfall, subplot, slider
    scatter_size : float
        size of points on plot
    linewidth : float
        linewidth of plot
    color : str
        color of plot
    offset : float
        offset between plot for waterfall plot
    rmv_baseline : bool
        whether or not to remove baseline (usually necessary for photoreflectance)
    subplot_values : list of float
        what values (from filename) should be included in the subplot
    return_plot : bool
        whether or not to return plot. It is necessary to return the plot in order to edit the plot such as titles or
        labels
    fit_variable : int
        which fit variable should be plotted. If this argument is passed only a plot of the fit variable will be
        returned
    identifier : str
        starting string of filename, used to filter out files in directory
    Returns
    -------
    x_data : list of list of float
        x data extracted from files
    y_data : list of list of float
        y data extracted from files
    fit : list of list of float
        fit values (this is only returned if a fit function is passed as an argument)

    """
    # Import files in directory
    filename, data = import_file(directory, pattern, identifier=identifier)

    # Convert filename to list of values rather than list of float
    filename = [n[0] for n in filename]

    # Initialize the variables
    x_data = []
    y_data = []

    # Extract relevant columns from data
    for n in data:
        x_data.append([i[x_col] for i in n])
        y_data.append([i[y_col] for i in n])

    # Generate auto offset
    if offset == "auto":
        offset = max(y_data[0])-min(y_data[0])

    # Remove baseline
    if rmv_baseline:

        # Function to plot polynomial
        def polyplot(x, deg):
            sum = 0
            for n in range(len(deg)):
                sum = sum + deg[n] * x ** (len(deg) - n - 1)
            return sum

        # Loop through y_data and remove baseline
        for n in range(len(y_data)):
            y_data[n] = y_data[n] - polyplot(np.asarray(x_data[n]), scipy.polyfit(x_data[n], y_data[n], 4))

    # If fit functions given generate guess data
    if fit_func != "none" and guess_func != "none":
        guess_data = guess_func(x_data, y_data, x_range)
        fit,_ = fit_all(x_data, y_data, guess_data, fit_func)

    # Generate colormap
    if color == 'range':

        # Generate colormap if color is a range
        # noInspection PyBroadException
        try:
            smin = min(filename)
            smax = max(filename)
            srange = smax - smin
            color_idx = [(n - smin) / srange for n in filename]
        except:
            color_idx = np.linspace(0, 1, len(filename))

    # Create Waterfall Plot
    if plot_type == "waterfall":

        for n in range(len(x_data)):

            # Plot data
            plt.scatter(x_data[n], np.asarray(y_data[n])+n*offset, s=scatter_size, color=cm.rainbow(color_idx[n]))

            # Create more points for smoother plot and plot data
            x_range = np.linspace(min(x_data[n]), max(x_data[n]), 1000)
            plt.plot(x_range, np.asarray(fit_func(x_range, *fit[n])) + n * offset, linewidth=linewidth, color='black')
        if not return_plot:
            plt.show()

    # Create Slider Plot
    if plot_type == "slider":

        # Initialize the figure
        fig = plt.figure()

        # Create Axes to plot
        plot_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
        slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])

        # Create slider
        slider = Slider(slider_ax, 'Offset', 0, len(filename) - 1, 0, valstep=1)

        def update(val):

            # Convert from value to index
            n = int(val)
            plot_ax.cla()
            plt.axes(plot_ax)
            # Plot data
            plt.scatter(x_data[n], np.asarray(y_data[n]) + n * offset, s=scatter_size, color=cm.rainbow(color_idx[n]))

            # Create more points for smoother plot and plot data
            x_range = np.linspace(min(x_data[n]), max(x_data[n]), 1000)
            plt.plot(x_range, np.asarray(fit_func(x_range, *fit[n])) + n * offset, linewidth=linewidth, color='black')
            plt.title(filename[n])
            fig.canvas.draw_idle()

        # Run update so that fig is not blank on generation
        update(0)

        # Run update when figure is changed
        slider.on_changed(update)
        plt.show()

    # Create Subplot Plot
    if plot_type == "subplot":

        # Filter out values if subplot values is included
        if subplot_values == "all":
            plots = [filename.index(n) for n in filename]
        else:
            plots = [filename.index(n) for n in subplot_values]

        # Generate starting variables
        fig, axes = plt.subplots(nrows=2, ncols=int((len(plots) + 1.5) / 2))

        # Loop over row in temp array and plot each
        for ax, n in zip(axes.flatten(), plots):
            a = 1
            b = 0.15

            plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
            plt.axes(ax)

            # Plot data
            plt.scatter(x_data[n], np.asarray(y_data[n]) + n * offset, s=scatter_size, color=cm.rainbow(color_idx[n]))

            # Create more points for smoother plot and plot data
            x_range = np.linspace(min(x_data[n]), max(x_data[n]), 1000)
            plt.plot(x_range, np.asarray(fit_func(x_range, *fit[n])) + n * offset, linewidth=linewidth, color='black')

        if not return_plot:
            plt.show()

    # Create plot of fit variable
    if fit_variable != "none":
        plt.cla()
        plt.scatter(filename, [n[fit_variable] for n in fit])
        if not return_plot:
            plt.show()

    if fit_func != "none" and guess_func != "none":
        return x_data, y_data, fit
    else:
        return x_data, y_data
