from json import load
from os.path import join
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
import cv2

BEATS = ['initial', 'final']


def convert_to_float(data):
    return np.array([float(val) for val in data])


def present_values(folder, map_type, params=None, return_plot=False, return_data=False):
    fig, ax = plt.subplots(ncols=len(BEATS), figsize=(10, 5))
    data =[]
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    for i, beat in enumerate(BEATS):
        path_to_results = join(folder, f"{beat}_{map_type}_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
        values = convert_to_float(sal_initial['values'])
        if params is not None:
            values = values[np.where(values != params)[0]]
        print(f"Mean value of {beat} beat: {np.nanmean(values)*100:.2f} +- {np.nanstd(values)*100:.2f}%")
        ax[i].hist(values[np.logical_not(np.isnan(values))]*100, bins=100)
        data.append(values[np.logical_not(np.isnan(values))]*100)
        ax[i].title.set_text(beat)
        ax[i].grid()
    plt.tight_layout()

    if return_plot:
        print("Returning PLOT (not data).")
        return ax
    
    if return_data:
        print("Returning DATA (not plot).")
        return data
    plt.show()


def present(folder, map_type, params=None, but_zeros=False, return_plot=False, return_data=False):
    fig, ax = plt.subplots(ncols=len(BEATS), figsize=(10, 5))
    data = []
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    for i, beat in enumerate(BEATS):
        path_to_results = join(folder, f"{beat}_{map_type}_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
        values = convert_to_float(sal_initial['values'])
        if params is not None:
            values = values[np.where(np.array(sal_initial[params[0]]) == params[1])[0]]
        if but_zeros:
            values = values[np.where(values != 0)[0]]
        print(f"Mean value of {beat} beat: {np.nanmean(values)*100:.2f} +- {np.nanstd(values)*100:.2f}%")
        ax[i].hist(values[np.logical_not(np.isnan(values))]*100, bins=100)
        data.append(values[np.logical_not(np.isnan(values))]*100)
        ax[i].title.set_text(beat)
        ax[i].grid()

    if return_plot:
        print("Returning PLOT (not data).")
        return ax

    if return_data:
        print("Returning DATA (not plot).")
        return data
    plt.show()


def get_accuracies(folder):
    for beat in BEATS:
        path_to_results = join(folder, f"{beat}_gb_grad_cam_map_metrics_v2.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
            print(f"{beat} beat: {len(np.where(np.array(sal_initial['pred_results']) == 'ok')[0]) / len(sal_initial['pred_results'])}")


def get_p_n(folder):
    results = {'initial': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}, 'mid': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}, 'final': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}}
    for i, beat in enumerate(BEATS):
        path_to_results = join(folder, f"{beat}_saliency_map_metrics.json")
        with open(path_to_results, 'r') as file:
            sal_initial = load(file)
        for i in range(len(sal_initial['pred_results'])):
            if sal_initial['pred_results'][i] == 'ok':
                if sal_initial['true_labels'][i] == 'normal':
                    results[beat]['tn'] += 1
                else:
                    results[beat]['tp'] += 1
            else:
                if sal_initial['true_labels'][i] == 'normal':
                    results[beat]['fp'] += 1
                else:
                    results[beat]['fn'] += 1
    return results

def calculate_metrics(folder):
    results = {'initial': {}, 'mid': {}, 'final': {}}
    res = get_p_n(folder)
    for beat in results.keys():
        print(res[beat])
        results[beat]['accuracy'] = (res[beat]['tp'] + res[beat]['tn']) / (res[beat]['tp'] + res[beat]['tn'] + res[beat]['fp'] + res[beat]['fn'])
        results[beat]['precision'] = res[beat]['tp'] / (res[beat]['tp'] + res[beat]['fp'])
    return results


def maps_comparison(beat, map_name, label, folder=f"."):
    path1 = join(folder, f"attribution_maps_no_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}")
    path2 = join(folder, f"attribution_maps_with_grid/attribution_maps/{map_name}/label_{beat}_beat/{label}")
    folder1 = listdir(path1)
    folder2 = listdir(path2)
    i = 0
    for file in folder1:
        if file in folder2:
            i += 1
            print(file)
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,10))
            ax1.imshow(cv2.imread(join(path1, file)))
            ax2.imshow(cv2.imread(join(path2, file)))
            ax1.axis('off')
            ax2.axis('off')
            plt.tight_layout()
            plt.show()
            if i == 20:
                break


def make_histogram(d):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure(figsize=(5, 5))
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa', density=True)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, fr'$\mu={np.mean(d)}, b={np.std(d)}$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


if '__main__' == __name__:
    res = get_p_n(r'.\XAI_metrics\metrics\no_grid')
    res = calculate_metrics(r'.\XAI_metrics\metrics\no_grid')
    print(res)
