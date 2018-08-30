# coding=utf-8
"""Il modulo SKLearn contiene tutte gli algoritmi di ML necessari"""
from __future__ import print_function
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="elementwise comparison failed")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")
warnings.filterwarnings("ignore", message="Variables are collinear")
warnings.filterwarnings("ignore", message="Precision and F-score are ill-defined")



import re
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix as conf_matr
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from subprocess import call
import os
from sklearn import tree as tr
import itertools
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import os.path

np.set_printoptions(threshold=np.nan)
import time


def LabelEncoder(dataset, target, header, risk_level, dataset_name, force_encoding=False):
    """
LabelEncoder effettua il Label Encoding del dataset e del target in ingresso
    :param dataset: il dataset di cui si vuole effettuare il Label Encoding
    :param target: il target di cui si vuole effettuare il Label Encoding
    :param risk_level: il livello per cui si vuole effettuare la classificazione
    :param dataset_name: il nome del dataset
    :param dataset_name: indica se si vuol forzare l'encoding del dataset, non tenendo conto delle codifiche precedenti già salvate sul file
    :return dataset: dataset in cui è stato effettuato il label Encoding
    :return target: target in cui è stato effettuato il label Encoding
    :return dataset_encoder: contiene un dictionary [index, label]
    :return target_encoder: contiene tutte le label del target
    """

    # Estraggo tutte le possibili Label dal dataset

    index_dict = 0
    labels = dict()
    column_index = 0
    directory = './labels/'
    filename_enc = directory + "encoding_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"
    filename_dataset = directory + "enc_data_" + risk_level + "_" + os.path.basename(dataset_name).split(".csv")[
        0] + "_" + str(len(dataset)) + ".npy"

    if os.path.exists(directory) == False:
        os.mkdir(directory)
    total_column = float(len(dataset[0]))

    date_i_1 = -1
    date_i_2 = -1
    date_i_3 = -1
    date_i_4 = -1

    for i in range(0, len(header)):
        if header[i] == 'DINZI_VLDT_GRZI':
            date_i_1 = i
        if header[i] == 'DFINE_VLDT_GRZI':
            date_i_2 = i
        if header[i] == 'data_prima_immatricolazione':
            date_i_3 = i
        if header[i] == 'data_ultima_voltura':
            date_i_4 = i

    if os.path.exists(filename_enc) == False or os.path.exists(filename_dataset) == False or force_encoding == True:
        for column in dataset.T:
            column_labels = dict()
            print("\rEncoding: " + str(int((column_index / total_column) * 100)) + "%", end="")
            tmp_labels = []
            for row in column:
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    row = to_num_date(row)
                if tmp_labels.__contains__(row) == False:
                    tmp_labels.append(row)
            tmp_labels = sorted(tmp_labels)
            for row in tmp_labels:
                column_labels[index_dict] = row
                labels[index_dict] = row
                index_dict = index_dict + 1
            for i in range(0, len(dataset)):
                if column_index == date_i_1 or column_index == date_i_2 or column_index == date_i_3 or column_index == date_i_4:
                    label = to_num_date(dataset[i, column_index])
                else:
                    label = dataset[i, column_index]
                dataset[i, column_index] = get_key(column_labels, label)
            column_index = column_index + 1
        print("\rEncoding: 100 %")
        dataset = np.array(dataset).astype(np.int)
        np.save(filename_enc, labels)
        np.save(filename_dataset, dataset)
    else:
        labels = np.load(filename_enc).item()
        dataset = np.load(filename_dataset)

    # Eseguo il Label Encoding per il target
    target_encoder = preprocessing.LabelEncoder()
    target_encoder.fit(target)
    target = target_encoder.transform(target)
    print("\rEncoding: 100%", end="")
    print("\nDone!")
    return dataset, np.array(target).astype(np.int), labels, target_encoder.classes_



def cross_validation(models, dataset, target, cv=1, confusion_matrix=False, statistics=False, labels=None, clfs_name=None,
                     risk_level=None, percentage=-1):
    """
cross_validation effettua la cross validation sul dataset specificato come parametro.
    :param model: specifica classificatore da utilizzare per la cross validation
    :param dataset
    :param target
    :param cv: specifica il numero di suddivisioni effettuate dalla cross validation
    :param confusion_matrix: bool value che indica se visualizzare o meno a video la matrice di confusione ad ogni iterazione della cross validation
    :param statistics: bool value che indica se visualizzare o meno a video precision, recall, f1-score e support ad ogni iterazione della cross validation
    :param labels: etichette di classe del target
    :param clf_name: nome del classificatore
    :param risk_level: livello di rischio per cui si sta effettuando la classificazione
    """

    print("\n##### CROSS VALIDATION RUN ########\n")

    with open("./statistics/statistics_" + risk_level + ".txt", "a") as myfile:
        myfile.write("####### CROSS VALIDATION START #######\n")
        if percentage==-1:
            myfile.write("Run oversampling without percentage\n\n\n")
            print("Run oversampling without percentage\n")
        else:
            myfile.write("Run oversampling with percentage for the less populous class: "+ str(percentage) +"%\n\n\n")
            print("Run oversampling with percentage for the less populous class: "+ str(percentage) + "%\n")
        myfile.close()
    if cv >= 2:
        cv_split = cv
    else:
        cv_split = 2


    start = time.time()
    y_test_tot = dict()
    predicted_tot = dict()
    kf = StratifiedKFold(n_splits=cv_split, random_state=None, shuffle=False).split(dataset, target)
    k = 1
    for train_index, test_index in kf:
        print("####### Fold " + str(k) + "/" + str(cv) + " #######")
        print("\nStart the training....")
        X_train, X_test = dataset[train_index], dataset[test_index]
        y_train, y_test = target[train_index], target[test_index]

        #Separo il training set e il test set
        if percentage == -1:
            X_train, y_train = ADASYN().fit_sample(X_train, y_train)
        else:
            index_B = []
            target_max = max(target)
            for i in range(0, len(y_train)):
                if y_train[i] == target_max:
                    index_B.append(i)

            count_in_train = 0
            for i in range(0, len(y_train)):
                if y_train[i] == target_max:
                    count_in_train = count_in_train + 1

            cB_perc = float(percentage)
            tot = float(len(y_train))
            cB_tot = int((cB_perc / 100.0) * tot) - count_in_train

            r_index_B = list()
            for i in range(0, cB_tot):
                r_index_B.append(random.randint(0, len(index_B) - 1))

            r_indexes_B = []
            for i in r_index_B:
                r_indexes_B.append(index_B[i])

            X_train = list(X_train) + list(X_train[r_indexes_B])
            y_train = list(y_train) + list(y_train[r_indexes_B])

        k = k + 1
        clfs_already_trained = []
        #Eseguo il training con i vari classificatori
        for model, name in zip(models, clfs_name):
            print("Train classifier " + name)
            clf = BaggingClassifier(model)
            clfs_already_trained.append(clf.fit(X_train, y_train))
        print("Training done!")
        all_predicted_clf = []
        for clf in clfs_already_trained:
            all_predicted_clf.append(clf.predict(X_test))

        index_dict = 0
        for n_of_clf in range(2, len(clfs_name)+1):
            predicted_clf = all_predicted_clf[:n_of_clf]
            for n_right in range(2, n_of_clf+1):

                y_test_tot[index_dict] = np.array([])
                predicted_tot[index_dict] = np.array([])

                print("Number of equal prediction between classification for less populous class: " + str(n_right))
                print("Number of classifier used: " + str(n_of_clf))

                predicted = []
                for i in range(0, len(predicted_clf[0])):
                    counter_right = 0
                    for j in range(0, n_of_clf):
                        if predicted_clf[j][i] == 1:
                            counter_right = counter_right+1
                    if(counter_right>= n_right):
                        predicted.append(1)
                    else:
                        predicted.append(0)
                acc = accuracy_score(y_test, predicted, normalize=True, sample_weight=None)
                print("Accuracy on the prediction: ", end="")
                print("%.2f" % acc)
                print("")

                if confusion_matrix == True:
                    cm = conf_matr(y_test, predicted)
                    print("\nConfusion matrix...")
                    print(pd.DataFrame(cm, index=labels, columns=labels))
                if statistics == True:
                    print("\nStatistics...")
                    print(classification_report(y_test, predicted, target_names=labels))

                y_test_tot[index_dict] = np.append(y_test_tot[index_dict], y_test)
                predicted_tot[index_dict] = np.append(predicted_tot[index_dict], predicted)
                index_dict = index_dict+1
                if cv == 1:
                    break

    end = time.time()
    index_dict = 0
    for n_of_clf in range(2, len(clfs_name) + 1):
        for n_right in range(2, n_of_clf + 1):
            y_test_tot_iter = y_test_tot[index_dict].astype(int)
            predicted_tot_iter = predicted_tot[index_dict].astype(int)

            with open("./statistics/statistics_" + risk_level + ".txt", "a") as myfile:
                myfile.write("Number of equal prediction between classification for less populous class: " + str(n_right) + "\n")
                myfile.write("Number of classifier used: " + str(n_of_clf) + "\n")
                myfile.write("\nElapsed times for cross validation: " + timer(start, end) + "\n\n")
                acc = accuracy_score(y_test_tot_iter, predicted_tot_iter, normalize=True, sample_weight=None)
                myfile.write("Accuracy on the prediction: ")
                print("%.2f" % acc, file=myfile)
                print("", file=myfile)
                myfile.write("Confusion Matrix\n")
                cm = conf_matr(y_test_tot_iter, predicted_tot_iter)
                print(pd.DataFrame(cm, index=labels, columns=labels), file=myfile)
                myfile.write("\n")
                print(classification_report(y_test_tot_iter, predicted_tot_iter, target_names=labels), file=myfile)
                myfile.write("\n\n")
                myfile.close()
            index_dict = index_dict+1


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def get_key(labels, label):
    for key, value in labels.items():
        if value == label:
            return key


def tree_as_pdf(dataset, target, features, path, labels):
    """
Elabora l'albero di decisione come pdf e lo salva nel path specificato
    :param dataset: dataset su cui eseguire il training
    :param target: target su cui eseguire il training
    :param features: attributi su cui si è basato l'albero di decisione
    :param path: dove salvare il file
   """

    if os.path.isdir("./dtree") == False:
        os.mkdir("./dtree")
    path = "./dtree/" + path

    print("Save the decision tree as pdf file into " + path)
    tree = DecisionTreeClassifier(max_depth=5).fit(dataset, target)
    tr.export_graphviz(tree,
                       out_file='temp.dot', feature_names=features)

    # Converte i numeri del .dot alle label corrette
    with open('./temp.dot', "r") as f:
        lines = f.readlines()
        f.close()
    with open('./temp.dot', "w") as f:
        for line in lines:
            split_gini = line.split(r"\ngini")
            if len(split_gini) > 1:
                number_gini = split_gini[0].split(" ")
                label = labels[int(float(number_gini[3]))]
                line = line.replace(number_gini[3], label)
            f.write(line + "\n")
        f.close

    call(['dot', '-Tpdf', 'temp.dot', '-o', path])
    print("Done!")
    os.remove('temp.dot')


def to_num_date(date):
    # input: formato 01JAN1990
    pattern = re.compile("[^0-31][0-31].*")
    if pattern.search(date) is None:
        return date
    months = dict()
    months['JAN'] = 1;
    months['FEB'] = 2;
    months['MAR'] = 3;
    months['APR'] = 4;
    months['MAY'] = 5;
    months['JUN'] = 6;
    months['JUL'] = 7;
    months['AUG'] = 8;
    months['SEP'] = 9;
    months['OCT'] = 10;
    months['NOV'] = 11;
    months['DEC'] = 12;
    month_str = str(date[2:5])
    day_str = date.split(month_str)
    day = day_str[0]
    year = day_str[1]
    conv_date = str(year + "/" + str(months[month_str]).zfill(2) + "/" + day)
    return conv_date


"""def MyLabelEncoder(dataset, target):

    #Estraggo tutte le possibili Label dal dataset
    print("\rEncoding: 0%", end="")
    index = 0
    total = float(len(dataset[0])*len(dataset))
    labels = []
    for column in dataset.T:
        print("\rEncoding: " + str(int((index / total) * 50)) + "%", end="")
        for row in column:
            if labels.__contains__(row) == False:
                labels.append(row)
            index = index + 1

    #Eseguo il Label Encoding per tutto il dataset
    dataset_encoder = preprocessing.LabelEncoder()
    dataset_encoder.fit(labels)

    total = float(len(dataset[0]))
    index = 0
    for column in dataset.T:
        print("\rEncoding: " + str(50 + int((index/total)*50)) + "%", end="")
        dataset[:, index] = dataset_encoder.transform(column)
        index = index + 1

    #Eseguo il Label Encoding per il target
    target_encoder = preprocessing.LabelEncoder()
    target_encoder.fit(target)
    target = target_encoder.transform(target)
    print("\rEncoding: 100%", end="")
    print("\nDone!")
    return np.array(dataset).astype(np.int), np.array(target).astype(np.int), dataset_encoder, target_encoder"""

"""def my_tree_as_pdf(dataset, target, features, path, labels):

    print("Save the decision tree as pdf file into " + path)
    tree = DecisionTreeClassifier().fit(dataset, target)
    tr.export_graphviz(tree,
                         out_file='temp.dot', feature_names=features)

    #Converte i numeri del .dot alle label corrette
    with open('./temp.dot', "r") as f:
        lines = f.readlines()
        f.close()
    with open('./temp.dot', "w") as f:
        for line in lines:
            split_gini = line.split(r"\ngini")
            if len(split_gini) > 1:
                number_gini = split_gini[0].split(" ")

                label = labels[int(float(number_gini[3]))]
                line = line.replace(number_gini[3], label)
            f.write(line+"\n")
        f.close

    call(['dot', '-Tpdf', 'temp.dot', '-o', path])
    print("Done!")
    #os.remove('temp.dot')"""

"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')"""

