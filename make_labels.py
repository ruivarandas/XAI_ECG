from convert_ECG_to_images import *

if __name__ == "__main__":
    folder = r'./data/mit-bih-arrhythmia-database-1.0.0'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i")
    parser.add_argument("-f")
    args = parser.parse_args()
    index = 0
    for file in range(int(args.i), int(args.f)):
        if file not in [110, 120, 204, 206, 211, 216, 218, 224, 225, 226, 227, 229]:
            signal, annotations = read_mit_database(folder, file)
            annotations.standardize_custom_labels()
            labels = annotations.symbol
            R_peaks = annotations.sample
            fs = annotations.fs
            new_folder = join(r'.\data\raw_figures', str(file))
            if not exists(new_folder):
                makedirs(new_folder)
            with open(join('./data/labels_mid', str(file)+'.txt'), 'w') as f:
                f.write("Sample\tLabel\n")
            with open(join('./data/labels_mid', str(file) + '.txt'), 'a') as f:
                if index < 5:
                    for i, segment in enumerate(labels[index:-5+index]):
                        f.write(str(i) + '\t' + labels[i+index] + '\n')
                else:
                    for i, segment in enumerate(labels[index:]):
                        f.write(str(i) + '\t' + labels[i+index] + '\n')
     