import os
import numpy as np
from read_data_210 import read_mit_database, generate_time, _ax_plot, normalize, argparse
import matplotlib.pyplot as plt
import json
import io
import cv2


def segment_ecg(signal, timestamps, num_cycles=5, heartbeat=5):
    timestamps = timestamps[1:]
    segments, limits = [], []
    inf = 100
    ind = num_cycles-heartbeat

    for i in range(0, len(timestamps)-num_cycles):
        aux = timestamps[i]
        first = i+num_cycles-ind
        second = i+num_cycles+1-ind
        if i < len(timestamps)-num_cycles - 1 and not timestamps[first]-inf < 0:
            right = timestamps[second] - aux
            left = timestamps[first] - aux
            top = np.max(signal[timestamps[first]-inf:timestamps[second]-inf])
            bottom = np.min(signal[timestamps[first]-inf:timestamps[second]-inf])
        elif timestamps[first]-inf < 0:
            right = timestamps[second]-inf
            left = timestamps[first]-inf
            top = np.max(signal[:right])
            bottom = np.min(signal[:right])
        else:
            right = timestamps[-1-ind] - aux
            left = timestamps[first] - aux
            top = np.max(signal[timestamps[first]-inf:])
            bottom = np.min(signal[timestamps[first]-inf:])
        if i < len(timestamps)-num_cycles - 1 and not timestamps[i]-inf < 0:
            segment = signal[timestamps[i]-inf:timestamps[i+num_cycles+1]-inf]
        elif timestamps[i]-inf < 0:
            segment = signal[:timestamps[i + num_cycles + 1] - inf]
        else:
            segment = signal[timestamps[i] - inf:]
        segments.append(segment)
        limits.append([left, right, bottom, top])

    return np.array(segments), np.array(limits)


def convert_values_to_pixels(x, y):
    """
    I made calibration lines in Excel based on the limits of the plot and the corresponding pixel positions
    :param x: int or float
    :param y: int or float
    :return:
    """
    return int(np.round(232.5*x + 375, 0)), int(np.round(-96.389*y + 227.5))


def config_labels():
    with open("config.json") as j:
        config = json.load(j)
        j.close()
    return config["labels_bin"]


def get_binary_label(la, labels_bin_list):
    if la in labels_bin_list['abnormal']:
        return 'abnormal'
    else:
        return 'normal'


def convert_to_binary(labels, bin_list):
    new_labels = []
    for label in labels:
        new_labels.append(get_binary_label(label, bin_list))
    return new_labels


def get_img_from_fig(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_pixels(fig, ax, x, y):
    xy_pixels = ax.transData.transform(np.vstack([x,y]).T)
    xpix, ypix = xy_pixels.T

    # In matplotlib, 0,0 is the lower left corner, whereas it's usually the upper
    # left for most image software, so we'll flip the y-coords...
    width, height = fig.canvas.get_width_height()
    ypix = height - ypix

    for xp, yp in zip(xpix, ypix):
        return int(xp), int(yp)


if __name__ == "__main__":
    folder = r'./data/mit-bih-arrhythmia-database-1.0.0'
    parser = argparse.ArgumentParser()
    parser.add_argument("-beat")
    args = parser.parse_args()
    beat = int(args.beat)
    binary_labels = config_labels()

    folders = []
    for file in set([file.split('.')[0] for file in os.listdir(folder)]):
        try:
            if int(file) in [100, 103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]:
                folders.append(int(file))
        except ValueError:
            pass

    with open(os.path.join(f"./ROI/{beat}_ROI.txt"), 'w') as f:
        f.write("Patient\tFile\tbottom\ttop\tleft\tright\tlabel\n")
        for file in folders:
            print(file)
            signal, annotations = read_mit_database(folder, file)
            annotations.standardize_custom_labels()
            if beat != 5:
                labels = convert_to_binary(annotations.symbol[beat+1:-5+beat], config_labels())
            else:
                labels = convert_to_binary(annotations.symbol[beat+1:], config_labels())
            R_peaks = annotations.sample
            fs = annotations.fs
            segmented_ecg, limits = segment_ecg(signal[:, 0], R_peaks, 5, beat)

            for i, segment in enumerate(segmented_ecg):
                print(f"{i}/{len(segmented_ecg)}", end='\r')
                time = generate_time(segment, fs)
                left, right, bottom, top = limits[i]

                # Transformation based on the plot (see the other file)
                left, right = left/fs - len(time)/(2*fs) + 5, right/fs - len(time)/(2*fs) + 5
                bottom, top = (bottom-np.mean(segment))*2/np.ptp(segment), (top-np.mean(segment))*2/np.ptp(segment)

                fig = plt.figure(figsize=(30, 4.5))
                ax = _ax_plot()
                time = generate_time(segment, fs)
                _ = plt.plot(np.array(time) - len(time)/(2*fs) + 5, normalize(segment), color='black')
                _ = plt.plot([left, right], [top, bottom], color=(55/255, 55/255, 55/255, 1), marker='.', linewidth=0)

                left, top = get_pixels(fig, ax, left, top)
                right, bottom = get_pixels(fig, ax, right, bottom)

                plt.close()

                f.write(f"{file}\t{i+1}_0\t{bottom}\t{top}\t{left}\t{right}\t{labels[i]}\n")
                
            print()
