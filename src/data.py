import zipfile
import os
import csv

def unzip(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()


if __name__ == "__main__":

    print('Creating output directories...')
    os.makedirs('../data/train/0', exist_ok=True)
    os.makedirs('../data/train/1', exist_ok=True)
    os.makedirs('../data/test', exist_ok=True)

    print('Decompressing files...')
    unzip('../data/train_labels.csv.zip', '../data')
    unzip('../data/train.zip', '../data/train')
    unzip('../data/test.zip', '../data/test')

    os.remove('../data/train_labels.csv.zip')
    os.remove('../data/train.zip')
    os.remove('../data/test.zip')
    os.remove('../data/sample_submissions.csv.zip')

    print('Moving images to correct folders...')
    with open('../data/train_labels.csv') as train_labels:
        csv_reader = csv.reader(train_labels, delimiter=',')
        next(csv_reader)
        for id, label in csv_reader:
            os.rename('../data/train/' + id + '.tif', '../data/train/' + label + '/' + id + '.tif')

    for img_test in os.listdir('../data/test/'):
        os.makedirs('../data/test/' + img_test[:-4] + '/', exist_ok=True)
        os.rename('../data/test/' + img_test, '../data/test/' + img_test[:-4] + '/' + img_test)

    os.remove('../data/train_labels.csv')
