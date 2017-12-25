import csv

# Code to add individual player stats to their respective teams and sum

list=[' 03-04',' 04-05',' 05-06',' 06-07',' 07-08',' 08-09',' 09-10',' 10-11',' 11-12',' 12-13',' 13-14',' 14-15' ]
# assuming Python 2
for i in range(len(list)):
    with open("Top10KeepersCleanSheets"+list[i]+".csv", "rb") as fp_in, open("TeamCleanSheets"+list[i]+".csv", "wb") as fp_out:
    with open("PlayersGoalScorersDistance"+list[i]+".csv", "rb") as fp_in, open("TeamGoalScorersDistance"+list[i]+".csv", "wb") as fp_out:
        reader = csv.reader(fp_in)
        writer = csv.writer(fp_out)
        r=[]
        for row in reader:
            r.append(row)

        for i in range(len(r)):
            for j in range(i+1,len(r)):

                if str(r[i][0])==str(r[j][0]):
                    for k in range(1,len(r[j])):
                        r[i][k]=float(r[i][k])+float(r[j][k])
                
            wr = csv.writer(fp_out, dialect='excel')
            wr.writerow(r[i])
