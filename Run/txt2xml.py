import xlwt
from os import listdir
from os.path import isfile, join

import os

def txt2xml(path_name):

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
        header_exel = ["Num_Fold", "Perc_Oversampling","Right_index","Num_Classifiers","Accuracy","Prec_T0","Rec_T0","F1_T0","Prec_T1","Rec_T1","F1_T1", "Run_Time"]
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
            else:
                sh.col(i).width = len(col) * 310

        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['yellow']
        style.pattern = pattern
        sh.write(3, 13, "Legenda", style)
        sh.write(4, 13, "Parametro", style)
        sh.write(4, 14, "Descrizione:",style)
        sh.write(5, 13, "Num_Fold:")
        sh.write(5, 14, "Indica il numero di fold utilizzati nella cross validation")
        sh.write(6, 13, "Perc_Oversampling:")
        sh.write(6, 14, "Indica la percentuale di elementi ripetuti nell'oversampling rispetto alla lunghezza del dataset iniziale")
        sh.write(7, 13, "Right_index:")
        sh.write(7, 14, "Indica il numero di classificatori minimo per classificare un entry con l'etichetta della classe meno popolosa")
        sh.write(8, 13, "Num_Classifiers:")
        sh.write(8, 14, "Indica il numero di classificatori minimo utilizzati")
        sh.write(9, 13, "Accuracy:")
        sh.write(9, 14, "Indica l'accuratezza")
        sh.write(10, 13, "Prec_X:")
        sh.write(10, 14, "Indica la precisione nella predizione del target X")
        sh.write(11, 13, "Rec_X:")
        sh.write(11, 14, "Indica il richiamo nella predizione del target X")
        sh.write(12, 13, "F1_X:")
        sh.write(12, 14, "Indica l'F1-Score nella predizione del target X")
        sh.write(13, 13, "Run_Time:")
        sh.write(13, 14, "Indica il momento in cui e' stato eseguito lo script per quella classificazione")
        sh.col(13).width = len("Perc_Oversampling")*400
        sh.col(14).width = 110*270

        with open(path) as f:
            lines = list(f.readlines())
            percentage = 0
            find_conf_mtr = 0
            find_stat_mtr = 0
            print_line = False
            index_excel = 0
            cv_iteration=0
            for line in lines:

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
                    sh.write(index_excel, 11, run_time)
        f.close()


    if len(onlyfiles)!= 0:
        book.save(filename_statistics)

