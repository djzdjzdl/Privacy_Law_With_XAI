import csv
import random
import numpy as np

class Fetch_Dataset:

    train_set = []
    test_set = []

    def __init__(self, file_path, number_of_string, number_of_label, split_proportion = 0.5):
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            Fetch_Dataset.Make_Train_Test(Fetch_Dataset.Dataset_Maker(csv_reader, number_of_string), split_proportion)

    #Get Every Labels
    def Dataset_Maker_With_Each_Label(csv_reader, number_of_string, number_of_label):
        tmp_datasets = []
        tmp_labels = []
        for row in csv_reader:
            if row[number_of_label] not in tmp_labels:
                tmp_labels.append(row[number_of_label])
            tmp_datasets += [ [row[number_of_string], tmp_labels.index(row[number_of_label]) ] ]
        random.shuffle(tmp_datasets)
        return tmp_datasets

    #Set Random Labels
    def Dataset_Maker(csv_reader, number_of_string):
        tmp_datasets = []
        for row in csv_reader:
            tmp_datasets += [ [row[number_of_string], random.randint(0, 1) ] ]
        random.shuffle(tmp_datasets)
        return tmp_datasets

    def Make_Train_Test(tmp_dataset, split_proportion):
        Fetch_Dataset.train_set, Fetch_Dataset.test_set = np.split(tmp_dataset, [int(split_proportion * len(tmp_dataset))])

    def Returner(self):
        return Fetch_Dataset.train_set, Fetch_Dataset.test_set

if __name__ == "__main__":
    train_set, test_set = Fetch_Dataset('./Resume.csv', 1, 3, 0.2).Returner()