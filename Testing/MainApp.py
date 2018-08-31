from __future__ import print_function
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")

import SKLearn
import ReadDataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

######## SETTARE QUI I PARAMETRI #########
dataset_path = 'dataset/itarea_compl2015_telematics_sent_clr_disc_y.csv' #path da dove leggere il csv
cross_validation_iteration=5 #Numero di iterazioni della cross validation
risk_levels = [
            'n_caused_claim',
            'NCD',
            'cost_caused_claim',
            'NNC',
] #Livelli di rischio da predire
#Classificatori ordinati in base al richiamo per ogni livello di rischio
classifiers_NCD = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            QuadraticDiscriminantAnalysis(),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            KNeighborsClassifier()
            ]

names_NCD = [
        "Naive Bayes", #0.79
        "Decision Tree", #0.76
        "Neural Net", #0.75
        "AdaBoost",#0.48
        "QDA", #0.46
        "Random Forest", #0.24
        "Nearest Neighbor" #0.18
]

classifiers_n_caused_claim = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            QuadraticDiscriminantAnalysis(),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            KNeighborsClassifier()
            ]

names_n_caused_claim = [
        "Naive Bayes", #0.79
        "Decision Tree", #0.74
        "Neural Net", #0.49
        "AdaBoost", #0.48
        "QDA", #0.43
        "Random Forest",  # 0.28,
        "Nearest Neighbor" #0.19
]

classifiers_cost_caused_claim = [
            GaussianNB(),
            DecisionTreeClassifier(max_depth=5),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            QuadraticDiscriminantAnalysis(),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            KNeighborsClassifier()
            ]

names_cost_caused_claim = [
        "Naive Bayes", #0.69
        "QDA",  # 0.71
        "Decision Tree", #0.62
        "Neural Net", #0.46
        "AdaBoost", #0.24,
        "Random Forest",  # 0.26,
        "Nearest Neighbor", #0.12
]


classifiers_NNC = [
            QuadraticDiscriminantAnalysis(),
            GaussianNB(),
            MLPClassifier(alpha=1),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier(),
            KNeighborsClassifier()
            ]

names_NNC = [
        "QDA", #0.81
        "Naive Bayes", #0.64
        "Neural Net", #0.62
        "Decision Tree", #0.57
        "Random Forest", #0.22
        "AdaBoost", #0.13
        "Nearest Neighbor" #0.12
]
##########################################



for risk_level in risk_levels:

    if risk_level == 'n_caused_claim':
        names = names_n_caused_claim
        classifiers = classifiers_n_caused_claim
    if risk_level == 'NNC':
        names = names_NNC
        classifiers = classifiers_NNC
    if risk_level == 'NCD':
        names = names_NCD
        classifiers = classifiers_NCD
    if risk_level == 'cost_caused_claim':
        names = names_cost_caused_claim
        classifiers = classifiers_cost_caused_claim

    decision_tree_path_pdf = 'dtree_' + risk_level + '.pdf'  # path dove verra' salvato l'albero di decisione
    print("####### SCRIPT FOR " + str.upper(risk_level) + " START #######\n")

    #print(ReadDataset.read_csv.__doc__)
    #Leggo il CSV e separo header, dataset e target(il target e' il livello di rischio di interesse)
    header, dataset, target = ReadDataset.read_csv(path_name=dataset_path, head=True, target=risk_level)


    #Trasformo il dataset e il target, cosi da avere solamente valori numerici e poter effettuare la classificazione
    #print(SKLearn.LabelEncoder.__doc__)
    dataset, target, dataset_encoder, target_encoder= SKLearn.LabelEncoder(dataset, target, header, risk_level=risk_level, dataset_name=dataset_path, force_encoding=False )

    #print(SKLearn.tree_as_pdf.__doc__)
    SKLearn.tree_as_pdf(dataset, target, features=header, path=decision_tree_path_pdf, labels=dataset_encoder)


    #print(SKLearn.cross_validation.__doc__)


    if risk_level == 'cost_caused_claim':
        targetA_012_B_3 = []
        # Divido target==0 da tutto il resto
        for i in range(0, len(target)):
            if target[i] != 3:
                targetA_012_B_3.append(0)
            else:
                targetA_012_B_3.append(1)

        SKLearn.cross_validation(models=classifiers, dataset=dataset, target=np.array(targetA_012_B_3),
                                     labels=['ClassA (range1,2,3)', 'ClassB (range 4)'], confusion_matrix=False,
                                     statistics=False, cv=cross_validation_iteration,
                                     clfs_name=names,# + " ( CV Iteration: " + str(cross_validation_iteration) + " )",
                                     risk_level=risk_level)
    else:
        SKLearn.cross_validation(models=classifiers, dataset=dataset, target=target,  labels=target_encoder, confusion_matrix=False, statistics=False, cv=cross_validation_iteration,
                                 clfs_name=names, # + " ( CV Iteration: " + str(cross_validation_iteration) + " )",
                                 risk_level=risk_level)


    for percentage in range(25, 55, 25):
        if risk_level == 'cost_caused_claim':
            targetA_012_B_3 = []
            # Divido target==0 da tutto il resto
            for i in range(0, len(target)):
                if target[i] != 3:
                    targetA_012_B_3.append(0)
                else:
                    targetA_012_B_3.append(1)

            SKLearn.cross_validation(models=classifiers, dataset=dataset, target=np.array(targetA_012_B_3),
                                     labels=['ClassA (range1,2,3)', 'ClassB (range 4)'], confusion_matrix=False,
                                     statistics=False, cv=cross_validation_iteration,
                                     clfs_name=names,
                                     # + " ( CV Iteration: " + str(cross_validation_iteration) + " )",
                                     risk_level=risk_level, percentage=percentage)
        else:
            SKLearn.cross_validation(models=classifiers, dataset=dataset, target=target, labels=target_encoder,
                                     confusion_matrix=False, statistics=False,
                                     cv=cross_validation_iteration,
                                     clfs_name=names,
                                     # + " ( CV Iteration: " + str(cross_validation_iteration) + " )",
                                     risk_level=risk_level, percentage=percentage)

