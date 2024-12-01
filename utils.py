from used_packages import *


#%%
def plot_signals(data, y_stacked,  title, knots=None, fsize=9, fsize_title=12, save_plot=False):
    # Set the global font size
    plt.rcParams.update({
        'font.size': fsize,          # Global font size
        'axes.titlesize': fsize,     # Title font size
        'axes.labelsize': fsize,     # Axis label font size
        'xtick.labelsize': fsize,     # X-axis tick font size
        'ytick.labelsize': fsize      # Y-axis tick font size
    })

    sensors = ['Body Accelerometer', 'Body Gyroscope', 'Total Accelerometer']
    axes = ['x', 'y', 'z']

    # Loop through each sensor and create a separate figure for each
    unique_labels = np.unique(y_stacked)
    for sensor_idx, (name, k) in enumerate(zip(sensors, [0, 3, 6])):
        fig, axs = plt.subplots(len(unique_labels), len(axes), figsize=(3 * len(axes), 2 * len(unique_labels)))
        plt.subplots_adjust(hspace=0.5)  # Adjust spacing between rows

        for i, label in enumerate(unique_labels):
            for j, axis in enumerate(axes):
                # Create the plot
                pd.DataFrame(data[np.where(y_stacked == label)[0], :, j + k].T).plot(
                    ax=axs[i, j],
                    legend=False
                )

                # Set the title
                axs[i, j].set_title(f'{axis}-axis, Label {label}')

        # Save each figure with a unique filename
        if knots:
            fig.suptitle(f'{title} {name} Sensor Data - {knots} knots', fontsize=fsize_title, y=0.93)
            if save_plot:
                plt.savefig(f'{name}_{title.lower()}_{knots}_knots.png', bbox_inches='tight', pad_inches=0.1)
        else:
            fig.suptitle(f'{title} {name} Sensor Data', fontsize=fsize_title, y=0.93)
            if save_plot:
                plt.savefig(f'{name}_{title.lower()}_signals.png', bbox_inches='tight', pad_inches=0.1)

        plt.show()



#%%

def plot_multiple_signals(data_raw, data_smooth, data_reduced, y_stacked, title, knots=None, fsize=9, fsize_title=12, save_plot=False):
    """
    Plots the signals for raw, smooth, and reduced data side-by-side for each axis and label.
    """
    # Set the global font size
    plt.rcParams.update({
        'font.size': fsize,
        'axes.titlesize': fsize,
        'axes.labelsize': fsize,
        'xtick.labelsize': fsize,
        'ytick.labelsize': fsize
    })

    sensors = ['Body Accelerometer', 'Body Gyroscope', 'Total Accelerometer']
    axes = ['x', 'y', 'z']
    datasets = {'Raw': data_raw, 'Smooth': data_smooth, 'Reduced': data_reduced}

    unique_labels = np.unique(y_stacked)

    # Loop through each sensor and create a separate figure for each
    for sensor_idx, (sensor_name, k) in enumerate(zip(sensors, [0, 3, 6])):
        for label in unique_labels:
            fig, axs = plt.subplots(len(axes), len(datasets), figsize=(4 * len(datasets), 3 * len(axes)))
            plt.subplots_adjust(hspace=0.4, wspace=0.2)  # Adjust spacing between rows and columns

            for i, axis in enumerate(axes):
                for j, (data_type, data) in enumerate(datasets.items()):
                    # Extract and plot the data for the given label, axis, and data type
                    pd.DataFrame(data[np.where(y_stacked == label)[0], :, i + k].T).plot(
                        ax=axs[i, j],
                        legend=False
                    )

                    # Set the title for each subplot
                    axs[i, j].set_title(f'{data_type} {axis}-axis - Label {label}')

            # Add a main title for the entire figure
            fig.suptitle(f'{title} {sensor_name} Sensor Data - Label {label}\nReduced Data {knots} knots', fontsize=fsize_title, y=0.95)
            if save_plot:
                plt.savefig(f'{sensor_name}_{title.lower()}_label-{label}_{knots}_knots.png', bbox_inches='tight', pad_inches=0.1)

            plt.show()

#%%

def raw_data_construct_Explore(str, subjects, GLOBALS):
    from collections import defaultdict
    raw = defaultdict()
    for num, i in enumerate(GLOBALS['sets']):

        for j in GLOBALS['axes']:

            for k, d in {'ba': 'body_acc', 'gyr': 'body_gyro', 'tot': 'tot_acc'}.items():

                locals()[f'X_raw_{k}_{j}_{i}'] = pd.concat([GLOBALS[f'{d}_{j}_{i}'], subjects],
                                                           axis=1).set_axis([l for l in range(0, 128)] + ['subject'], axis=1)
    for subject in np.unique(subjects):
        for label in np.unique(GLOBALS[f'label_{str}']):
            for sensor in GLOBALS['sensors']:
                X_sensor = []
                for axis in GLOBALS['axes']:
                    X_axis = []
                    X_fltr = locals()[f'X_raw_{sensor}_{axis}_{str}'].iloc[np.where(GLOBALS[f'label_{str}'] == label)[0], :]
                    X_fltr = X_fltr[X_fltr['subject'] == subject]
                    # iterate over rows, extract non-overlapping columns
                    for row in X_fltr.iloc[:-1, :].itertuples():
                        X_axis.append(row[1:65])
                    X_axis.append(X_fltr.iloc[-1, :-1])
                    X_axis = np.array([i for list in X_axis for i in list])
                    X_sensor.append(X_axis)
                raw[f'Raw_{sensor}_XYZ_subj{subject}_{label}_{str}'] = np.stack(X_sensor, axis=1)
    return dict(raw)

#%%


def raw_data_construct_ML(GLOBALS):
    local_vars = {}
    # add subject numbers to raw window data
    sensors = ['ba', 'gyr', 'tot']
    axes = ['x', 'y', 'z']
    sets = ['train', 'test']
    subjects = {0: GLOBALS['Str'], 1: GLOBALS['Sts']}
    for num, i in enumerate(sets):

        for j in axes:

            for k, d in {'ba': 'body_acc', 'gyr': 'body_gyro', 'tot': 'tot_acc'}.items():

                locals()[f'X_raw_{k}_{j}_{i}'] = pd.concat([GLOBALS[f'{d}_{j}_{i}'], subjects[num]],
                                                            axis=1).set_axis([l for l in range(0, 128)] + ['subject'], axis=1)

    def raw_data_construct2(str, subjects=subjects, LOCALS=locals()):
        from collections import defaultdict
        raw = defaultdict()
        for subject in np.unique(subjects):
            for label in np.unique(GLOBALS[f'label_{str}']):
                for sensor in sensors:
                    X_sensor = []
                    for axis in axes:
                        X_axis = []
                        X_fltr = LOCALS[f'X_raw_{sensor}_{axis}_{str}'].iloc[np.where(GLOBALS[f'label_{str}'] == label)[0], :]
                        X_fltr = X_fltr[X_fltr['subject'] == subject]
                        # iterate over rows, extract non-overlapping columns
                        for row in X_fltr.iloc[:-1, :].itertuples():
                            X_axis.append(row[1:65])
                        X_axis.append(X_fltr.iloc[-1, -129:-65])
                        X_axis.append(X_fltr.iloc[-1, -65:-1])
                        raw[f'Raw_{sensor}_XYZ_subj{subject}_{label}_{axis}_{str}'] = np.stack(X_axis, axis=1)

        return dict(raw)


    RAW_train2 = raw_data_construct2('train', subjects[0][0])
    RAW_test2 = raw_data_construct2('test', subjects[1][0])

    pattern_sens = f"_({'|'.join(sensors)})_"
    pattern_set = f"_({'|'.join(sets)})"

    for set in sets:
        for key, value in locals()[f'RAW_{set}2'].items():
            ax = re.findall(r'_([xyz])_', key)[0]
            label = re.findall(r'_([0-9]+)_', key)[0]
            sens = re.findall(pattern_sens, key)[0]
            se = re.findall(pattern_set, key)[0]

            locals()[f'RAW_{sens}_{ax}_{se}'] = []
            locals()[f'labels_{sens}_{ax}_{se}'] = []



        for key, value in locals()[f'RAW_{set}2'].items():
            ax = re.findall(r'_([xyz])_', key)[0]
            label = re.findall(r'_([0-9]+)_', key)[0]
            sens = re.findall(pattern_sens, key)[0]
            se = re.findall(pattern_set, key)[0]

            locals()[f'RAW_{sens}_{ax}_{se}'].append(value.T)
            locals()[f'labels_{sens}_{ax}_{se}'].append([label]*value.shape[1])

    raw_data = defaultdict()
    raw_labels = defaultdict()

    for set in sets:
        locals()[f'X_raw_{set}'] = []
        locals()[f'y_raw_{set}'] = []
        for sensor in sensors:
            for axis in axes:
                locals()[f'RAW_{sensor}_{axis}_{set}'] = np.concatenate(locals()[f'RAW_{sensor}_{axis}_{set}'], axis=0)
                locals()[f'labels_{sensor}_{axis}_{set}'] = np.concatenate(locals()[f'labels_{sensor}_{axis}_{set}'], axis=0)

                locals()[f'X_raw_{set}'].append(locals()[f'RAW_{sensor}_{axis}_{set}'])
                locals()[f'y_raw_{set}'] = locals()[f'labels_{sensor}_{axis}_{set}']

        locals()[f'X_raw_{set}'] = np.stack(locals()[f'X_raw_{set}'], axis=2)
        raw_data[f'X_raw_{set}'] = locals()[f'X_raw_{set}']
        raw_labels[f'y_raw_{set}'] = locals()[f'y_raw_{set}']

    return raw_data, raw_labels

#%%
def BS_partition_unity(GLOBALS, save_plot=False):

    plt.figure(figsize=(10, 3), dpi=100)
    plt.plot(GLOBALS['B1'])
    plt.vlines(x=10, ymin=0, ymax=0.65, linestyles='dashed', color='lightblue')
    plt.hlines(y=0.045, xmin=-2.5, xmax=11.5, linestyles='dashed', color='lightgray')
    plt.hlines(y=0.37, xmin=-2.5, xmax=11.5, linestyles='dashed', color='lightgray')
    plt.hlines(y=0.54, xmin=-2.5, xmax=11.5, linestyles='dashed', color='lightgray')
    plt.plot(10,0.045, marker='o', markersize=5, markerfacecolor='black', markeredgewidth=0)
    plt.plot(10,0.37, marker='o', markersize=5, markerfacecolor='black', markeredgewidth=0)
    plt.plot(10,0.54, marker='o', markersize=5, markerfacecolor='black', markeredgewidth=0)
    plt.text(10.5, 0.07, '2x0.045', fontsize=12, color='black')
    plt.text(10.5, 0.29, '0.37', fontsize=12, color='black')
    plt.text(10.5, 0.57, '0.54', fontsize=12, color='black')
    plt.text(10.5, 0.8, '0.045 + 0.045 + 0.37 + 0.54 = 1', fontsize=12, color='red')

    plt.title(f'B-Spline Basis\n{3}rd degree polynomial w/ {GLOBALS["knots"]} knots\nColumns add up to 1', fontsize=13)
    plt.xlim(left=-3)
    if save_plot:
        plt.savefig(f'BS_partition_of_unity.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()