from __future__ import print_function
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Duplicate key in file")

import SKLearn
import ReadDataset
import txt2xml

######## SETTARE QUI I PARAMETRI #########
csv_path = 'dataset/itarea_compl2016_telematics_sent_clr_disc_y.csv' #path da dove leggere il csv
cross_validation_iteration=5 #Numero di iterazioni della cross validation
risk_levels = [ #Livelli di rischio da utilizzare
            'n_caused_claim',
            #'NCD',
            #'cost_caused_claim',
            #'NNC',
            ]
##########################################



for risk_level in risk_levels:
    print("####### SCRIPT FOR " + str.upper(risk_level) + " START #######\n")

    #print(ReadDataset.read_csv.__doc__)
    #Leggo il CSV e separo header, dataset e target(il target e' il livello di rischio di interesse)
    header, dataset, target = ReadDataset.read_csv(path_name=csv_path, head=True, target=risk_level)


    #Trasformo il dataset e il target, cosi da avere solamente valori numerici e poter effettuare la classificazione
    #print(SKLearn.LabelEncoder.__doc__)
    dataset, target, dataset_encoder, target_encoder= SKLearn.LabelEncoder(dataset, target, header, risk_level=risk_level, dataset_name=csv_path, force_encoding=False )

    #print(SKLearn.tree_as_pdf.__doc__)
    SKLearn.tree_as_pdf(dataset, target, features=header, risk_level=risk_level, labels=dataset_encoder)


    SKLearn.cross_validation(dataset=dataset,
                             target=target,
                             labels=target_encoder,
                             confusion_matrix=True,
                             statistics=True,
                             cv=cross_validation_iteration,
                             risk_level=risk_level,
                             percentage=25)
txt2xml.txt2xml()