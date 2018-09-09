import xlwt
from os import listdir
from os.path import isfile, join

import os

def txt2xls(path_name):
    c1 = 25
    c2 = 26
    year = path_name[len(path_name) - 35:len(path_name) - 31]
    folder_statistics = "./statistics_" + year + "/"
    filename_xls = "statistics_" + year + ".xls"
    filename_statistics = folder_statistics + filename_xls

    onlyfiles = list([f for f in listdir(folder_statistics) if isfile(join(folder_statistics, f))])
    book = xlwt.Workbook()


    if onlyfiles.__contains__(filename_xls) == True:
        os.remove(filename_statistics)
        onlyfiles.remove(filename_xls)

    for file in onlyfiles:
        if file.__contains__(".txt") == False:
            onlyfiles.remove(file)

    for filename in onlyfiles:
        path = folder_statistics+filename
        header_exel = ["Num_Fold", "Perc_Oversampling","Right_index","Num_Classifiers","Accuracy","Prec_T0","Rec_T0","F1_T0","Prec_T1","Rec_T1","F1_T1","TP", "TN", "FP", "FN","Run_Time", "N_Cluster", "Clf_Name", "Bagging", "Max_features", "Max_samples", "Num_estimator"]
        filename_split = filename.split(".txt")
        sheet_name = filename_split[0]

        sh = book.add_sheet(sheet_name[16:])
        style = xlwt.XFStyle()
        font = xlwt.Font()
        font.bold = True
        style.font = font
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['ice_blue']
        style.pattern = pattern
        for i, col in enumerate(header_exel):
            sh.write(0, i, col, style=style)
            if col == "Run_Time":
                sh.col(i).width = len(col)*500
            elif col == "TP" or col=="TN" or col == "FP" or col == "FN":
                sh.col(i).width = len(col) * 1000
            else:
                sh.col(i).width = len(col) * 310

        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
        style.pattern = pattern

        sh.write(2, c1, "Attenzione: il valore -1 indica che in questa fase di test non e' stata utilizzato quel parametro.")
        sh.write(3, c1, "Legenda", style)
        sh.write(4, c1, "Parametro", style)
        sh.write(4, c2, "Descrizione:",style)
        sh.write(5, c1, "Num_Fold:")
        sh.write(5, c2, "Indica il numero di fold utilizzati nella cross validation")
        sh.write(6, c1, "Perc_Oversampling:")
        sh.write(6, c2, "Indica la percentuale di elementi ripetuti nell'oversampling rispetto alla lunghezza del dataset iniziale")
        sh.write(7, c1, "Right_index:")
        sh.write(7, c2, "Indica il numero di classificatori minimo per classificare un entry con l'etichetta della classe meno popolosa")
        sh.write(8, c1, "Num_Classifiers:")
        sh.write(8, c2, "Indica il numero di classificatori minimo utilizzati")
        sh.write(9, c1, "Accuracy:")
        sh.write(9, c2, "Indica l'accuratezza")
        sh.write(10, c1, "Prec_X:")
        sh.write(10, c2, "Indica la precisione nella predizione del target X")
        sh.write(11, c1, "Rec_X:")
        sh.write(11, c2, "Indica il richiamo nella predizione del target X")
        sh.write(12, c1, "F1_X:")
        sh.write(12, c2, "Indica l'F1-Score nella predizione del target X")
        sh.write(13, c1, "TP:")
        sh.write(13, c2, "Veri Positivi")
        sh.write(14, c1, "TN:")
        sh.write(14, c2, "Veri Negativi")
        sh.write(15, c1, "FP:")
        sh.write(15, c2, "Falsi Positivi")
        sh.write(16, c1, "FN:")
        sh.write(16, c2, "Falsi Negativi")
        sh.write(17, c1, "Run_Time:")
        sh.write(17, c2, "Indica il momento in cui e' stato eseguito lo script per quella classificazione")
        sh.write(18, c1, "N_Cluster:")
        sh.write(18, c2, "Indica il numero di cluster utilizzati in fase di oversampling")
        sh.write(19, c1, "Clf_Name:")
        sh.write(19, c2, "Classificatore utilizzato")
        sh.write(20, c1, "Bagging:")
        sh.write(20, c2, "Indica se e' stato utilizzato l'operatore di bagging")
        sh.write(21, c1, "Max_features:")
        sh.write(21, c2, "Parametro di bagging")
        sh.write(22, c1, "Max_samples:")
        sh.write(22, c2, "Parametro di bagging")
        sh.write(23, c1, "Num_Estimator:")
        sh.write(23, c2, "Parametro di bagging")

        sh.col(c1).width = len("Perc_Oversampling")*400
        sh.col(c2).width = 110*270

        with open(path) as f:
            n_cluster = -1
            max_f1=-1
            max_n_right=-1
            max_n_clf=-1
            max_perc=-1
            max_row=-1
            index_right = -1
            n_clf = -1
            max_features = -1
            max_samples = -1
            max_n_cluster=-1
            lines = list(f.readlines())
            percentage = -1
            find_conf_mtr = 0
            find_stat_mtr = 0
            print_line = False
            index_excel = 0
            cv_iteration=-1
            n_estimator=1
            clf_name=""
            bagging = ""
            for line in lines:

                if line.__contains__("Num Estimator: "):
                    l_split = line.split(": ")
                    n_estimator = l_split[1]

                if line.__contains__("Max Features: "):
                    l_split = line.split(": ")
                    max_features = l_split[1]
                if line.__contains__("Max Samples: "):
                    l_split = line.split(": ")
                    max_samples = l_split[1]

                if line.__contains__("Algorithm name: "):
                    l_split = line.split(": ")
                    clf_name = l_split[1]

                if line.__contains__("Bagging: "):
                    l_split = line.split(": ")
                    bagging = l_split[1]

                if line.__contains__("umber of cluster: "):
                    l_split = line.split(": ")
                    n_cluster = l_split[1]
                if line.__contains__("SCRIPT RUN "):
                    l_split = line.split("SCRIPT RUN ")
                    run_time = l_split[1]
                if line.__contains__("validation fold"):
                    fold_num_split = line.split(": ")
                    cv_iteration = int(fold_num_split[1])
                if line.__contains__("Run oversampling without per"):
                    percentage=0
                elif line.__contains__("Run oversampling with per"):
                    with_perc = line.split(": ")
                    number = with_perc[1].split("%")
                    percentage = int(number[0])
                if line.__contains__("Number of equal prediction between classification for"):
                    l_split = line.split(": ")
                    index_right = int(l_split[1])
                if line.__contains__("Number of classifier"):
                    l_split = line.split(": ")
                    n_clf = int(l_split[1])
                if line.__contains__("Accuracy on the"):
                    l_split = line.split(": ")
                    accuracy = float(l_split[1])

                ####### LETTURA MATRICE CONFUSIONE #########
                if filename.__contains__("cost"):
                    if find_conf_mtr == 4:
                        find_conf_mtr = 0
                    if find_conf_mtr == 3:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(4, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x10 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x11 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1

                    if find_conf_mtr == 2:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(4, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x00 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x01 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1
                    if find_conf_mtr == 1:
                        find_conf_mtr=find_conf_mtr+1
                    if line.__contains__("Confus"):
                        find_conf_mtr=find_conf_mtr+1
                else:
                    if find_conf_mtr == 4:
                        find_conf_mtr = 0
                    if find_conf_mtr == 3:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(2, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x10 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x11 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1

                    if find_conf_mtr == 2:
                        l_split = line.split(" ")
                        first_conf_matrix = True
                        for i in range(2, len(l_split)):
                            if l_split[i] != '':
                                if first_conf_matrix == True:
                                    x00 = int(l_split[i])
                                    first_conf_matrix = False
                                elif first_conf_matrix == False:
                                    x01 = int(l_split[i])
                        find_conf_mtr = find_conf_mtr + 1
                    if find_conf_mtr == 1:
                        find_conf_mtr = find_conf_mtr + 1
                    if line.__contains__("Confus"):
                        find_conf_mtr = find_conf_mtr + 1
                    #############################################

                if filename.__contains__("cost") == True:
                    if find_stat_mtr == 4:
                        print_line = True
                        find_stat_mtr = 0
                    if find_stat_mtr == 3:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(7, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s10 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s11 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1

                    if find_stat_mtr == 2:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(7, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s00 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s01 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1
                    if find_stat_mtr == 1:
                        find_stat_mtr = find_stat_mtr + 1
                    if line.__contains__("precisi"):
                        find_stat_mtr = find_stat_mtr + 1
                else:
                    if find_stat_mtr == 4:
                        print_line = True
                        find_stat_mtr = 0
                    if find_stat_mtr == 3:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(15, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s10 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s11 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1

                    if find_stat_mtr == 2:
                        l_split = line.split(" ")
                        first_stat_matrix = True
                        for i in range(15, len(l_split)):
                            if l_split[i] != '':
                                if first_stat_matrix == True:
                                    s00 = float(l_split[i])
                                    first_stat_matrix = False
                                elif first_stat_matrix == False:
                                    s01 = float(l_split[i])
                                    break
                        find_stat_mtr = find_stat_mtr + 1
                    if find_stat_mtr == 1:
                        find_stat_mtr = find_stat_mtr + 1
                    if line.__contains__("precisi"):
                        find_stat_mtr = find_stat_mtr + 1

                if print_line == True:
                    print_line = False
                    #header = "Right index,Num Classifiers,Accuracy,Prec T0,Rec T0,F1 T0,Prec T1,Rec T1,F1 T1"
                    if percentage == 0:
                        percentage_p = "No"
                    else:
                        percentage_p = str(percentage) + "%"
                    if float(s00+s01) > 0.0:
                        f11 = "{0:.2f}".format(float(s00*s01*2)/float(s00+s01))
                    else:
                        f11 = 0.0
                    if float(s10+s11) > 0.0:
                        f12 = "{0:.2f}".format(float(s10*s11*2)/float(s10+s11))
                    else:
                        f12 = 0.0

                    index_excel = index_excel + 1
                    sh.write(index_excel, 0, cv_iteration)
                    sh.write(index_excel, 1, percentage_p)
                    sh.write(index_excel, 2, index_right)
                    sh.write(index_excel, 3, n_clf)
                    sh.write(index_excel, 4, accuracy)
                    sh.write(index_excel, 5, s00)
                    sh.write(index_excel, 6, s01)
                    sh.write(index_excel, 7, f11)
                    sh.write(index_excel, 8, s10)
                    sh.write(index_excel, 9, s11)
                    sh.write(index_excel, 10, f12)
                    sh.write(index_excel, 11, x00)
                    sh.write(index_excel, 12, x11)
                    sh.write(index_excel, 13, x01)
                    sh.write(index_excel, 14, x10)
                    sh.write(index_excel, 15, run_time)
                    sh.write(index_excel, 16, n_cluster)
                    sh.write(index_excel, 17, clf_name)
                    sh.write(index_excel, 18, bagging)
                    sh.write(index_excel, 19, max_features)
                    sh.write(index_excel, 20, max_samples)
                    sh.write(index_excel, 21, n_estimator)
                    if float(f12)>=float(max_f1):
                        max_f1=float(f12)
                        max_perc=percentage_p
                        max_n_right=index_right
                        max_n_clf=n_clf
                        max_row = index_excel+1
                        max_n_cluster = n_cluster
                        max_bagging = bagging
                        max_clf_name = clf_name


        f.close()
        sh.write(25, c1, "Risultato migliore", style=style)
        sh.write(26, c1, "Riga " + str(max_row)+ ". Si ha un F1-Score="+str(int(max_f1*100))+"%.")
        sh.write(27, c1, "Prametri utilizzati: n_right="+str(max_n_right)+", n_clf="+str(max_n_clf)+", percentage=" + str(max_perc) +", n_cluster=" + str(max_n_cluster))
        sh.write(28, c1, "                     bagging="+ str(max_bagging) + " e classificatore=" + str(max_clf_name))
        sh.write(29, c1, "Entry della classe A: " + str(x00 + x01))
        sh.write(30, c1, "Entry della classe B: " + str(x10 + x11))

    if len(onlyfiles)!= 0:
        book.save(filename_statistics)
