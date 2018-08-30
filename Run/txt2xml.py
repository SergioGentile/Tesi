import xlwt
from os import listdir
from os.path import isfile, join

import os

def txt2xml():
    directory = './statistics/'
    onlyfiles = list([f for f in listdir(directory) if isfile(join(directory, f))])
    book = xlwt.Workbook()
    filename_xls = 'statistics.xls'
    if onlyfiles.__contains__(filename_xls) == True:
        os.remove(directory + filename_xls)
        onlyfiles.remove(filename_xls)
    for filename in onlyfiles:
        path = directory+filename
        header_exel = ["Perc_Oversampling","Right_index","Num_Classifiers","Accuracy","Prec_T0","Rec_T0","F1_T0","Prec_T1","Rec_T1","F1_T1"]
        filename_split = filename.split(".txt")
        sheet_name = filename_split[0]

        sh = book.add_sheet(sheet_name)
        style = xlwt.XFStyle()
        # font
        font = xlwt.Font()
        font.bold = True
        style.font = font
        pattern = xlwt.Pattern()
        pattern.pattern = xlwt.Pattern.SOLID_PATTERN
        pattern.pattern_fore_colour = xlwt.Style.colour_map['ice_blue']
        style.pattern = pattern
        for i, col in enumerate(header_exel):
            sh.write(0, i, col, style=style)
            sh.col(i).width = len(col)*290

        with open(path) as f:
            lines = list(f.readlines())
            percentage = 0
            find_conf_mtr = 0
            find_stat_mtr = 0
            print_line = False
            index_excel = 0
            for line in lines:

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
                    sh.write(index_excel, 0, percentage_p)
                    sh.write(index_excel, 1, index_right)
                    sh.write(index_excel, 2, n_clf)
                    sh.write(index_excel, 3, accuracy)
                    sh.write(index_excel, 4, s00)
                    sh.write(index_excel, 5, s01)
                    sh.write(index_excel, 6, f11)
                    sh.write(index_excel, 7, s10)
                    sh.write(index_excel, 8, s11)
                    sh.write(index_excel, 9, f11)
        f.close()

    book.save(directory + filename_xls)

