from __future__ import print_function
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")
warnings.filterwarnings("ignore", message="numpy.core.umath_tests")
import SKLearn
import ReadDataset
import txt2xls
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

######## SETTARE QUI I PARAMETRI #########
dataset_paths = [
    'dataset/itarea_compl2016_telematics_sent_clr_disc_y.csv',
    #'dataset/itarea_compl2015_telematics_sent_clr_disc_y.csv'
    ] #path da dove leggere il csv di Reale Mutua

n_fold=5 #Numero di fold utilizzati nella cross validation
risk_levels = [ #Livelli di rischio da utilizzare
            #'n_caused_claim',
            'NCD',
            'cost_caused_claim',
            'NNC',
            ]

classifiers= [
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        AdaBoostClassifier(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        KNeighborsClassifier()
    ]

names_classifiers = [
        "Naive Bayes",
        "QDA",
        "AdaBoost",
        "Decision Tree",
        "Neural Net",
        "Random Forest",
        "Nearest Neighbor",
    ]

##########################################

for dataset_path in dataset_paths:
    for risk_level in risk_levels:
        print("####### SCRIPT FOR " + str.upper(risk_level) + " START #######\n")

        #print(ReadDataset.read_csv.__doc__)
        #Leggo il CSV e separo header, dataset e target, dove il target e' il livello di rischio di interesse.
        #Dal dataset vengono eliminato gli altri tre livelli di rischio.
        header, dataset, target = ReadDataset.read_csv(path_name=dataset_path, target=risk_level)


        #Trasformo il dataset e il target, cosi da avere solamente valori numerici e poter effettuare la classificazione
        dataset, target, dataset_encoder, target_encoder= SKLearn.LabelEncoder(dataset, target, header, risk_level=risk_level, dataset_name=dataset_path, force_encoding=False )

        #Stampa (formato pdf) l'albero di decisione utilizzato nella classificazione del livello di rischio passato come parametro
        #SKLearn.tree_as_pdf(dataset=dataset, target=target, features=header, risk_level=risk_level, labels=dataset_encoder, dataset_name=dataset_path)


        for model, model_name in zip(classifiers, names_classifiers):
            for max_features in np.linspace(0.1, 1, 5):
                for max_samples in np.linspace(0.1, 1, 5):
                    SKLearn.cross_validation(model,
                                     model_name,
                                     dataset=dataset,
                                     target=target,
                                     dataset_name=dataset_path,
                                     labels=target_encoder,
                                     n_fold=n_fold,
                                     confusion_matrix=False,
                                     statistics=False,
                                     risk_level=risk_level,
                                     bagging=True,
                                     max_features=max_features,
                                     max_samples=max_samples,
                                     n_estimator=7
                                     )
        for model, model_name in zip(classifiers, names_classifiers):
            SKLearn.cross_validation(model,
                                     model_name,
                                     dataset=dataset,
                                     target=target,
                                     dataset_name=dataset_path,
                                     labels=target_encoder,
                                     n_fold=n_fold,
                                     confusion_matrix=False,
                                     statistics=False,
                                     risk_level=risk_level,
                                     bagging=False
                                     )

    #Trasforma i file txt rilasciati dalla cross validation in excel cosi' da esser letti piu' facilmente.
    txt2xls.txt2xls(path_name=dataset_path)