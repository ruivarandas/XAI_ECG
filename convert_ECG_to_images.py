import numpy as np
from os.path import join, exists
from os import makedirs, listdir
import matplotlib.pylab as plt
from matplotlib.ticker import AutoMinorLocator
from biosignalsnotebooks import generate_time


def read_mit_database(path, signal):
    import wfdb

    record = wfdb.rdrecord(join(path, str(signal)))
    annotation = wfdb.rdann(join(path, str(signal)), 'atr')

    return np.array(record.p_signal), annotation


def segment_ecg(signal, timestamps, num_cycles=5):
    segments = []
    inf = 100
    for i in range(0, len(timestamps)-num_cycles):
        if i < len(timestamps)-num_cycles - 1 and not timestamps[i]-inf < 0:
            segments.append(signal[timestamps[i]-inf:timestamps[i+num_cycles+1]-inf])
        elif timestamps[i]-inf < 0:
            segments.append(signal[:timestamps[i + num_cycles + 1] - inf])
        else:
            segments.append(signal[timestamps[i] - inf:])
    return np.array(segments)


def normalize(signal):
    norm = signal - np.mean(signal)
    norm = norm/np.ptp(norm)
    return norm*2


def _ax_plot(secs=10):
    ax = plt.axes()
    ax.set_ylim(-1.8, 1.8)
    ax.set_xlim(0, secs)
    ax.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)
    return ax


if __name__ == '__main__':
    # Path to database
    folder = r'./data/mit-bih-arrhythmia-database-1.0.0'

    folders = []
    for file in set([file.split('.')[0] for file in listdir(folder)]):
        try:
            folders.append(int(file))
        except ValueError:
            pass

    for file in folders:
        print(file)
        signal, annotations = read_mit_database(folder, file)
        annotations.standardize_custom_labels()
        labels = annotations.symbol
        R_peaks = annotations.sample
        fs = annotations.fs
        segmented_ecg = segment_ecg(signal, R_peaks, 5)
        new_folder = join(r'./data/raw_figures_no_grid', str(file))
        if not exists(new_folder):
            makedirs(new_folder)
        for i, segment in enumerate(segmented_ecg):
            print(f"{i}/{len(segmented_ecg)}", end='\r')
            fig = plt.figure(figsize=(30, 4.5))
            _ = _ax_plot()
            time = generate_time(segment[:, 0], fs)
            _ = plt.plot(np.array(time) - len(time)/(2*fs) + 5, normalize(segment[:, 0]), color='black')
            plt.show()
            fig.savefig(join(new_folder, str(i) + '_' + str(0)))
            plt.close(fig)
