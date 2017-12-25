import csv

# Code to merge files to generate iterative training data -> 2003 season, 2003-2004 season, 2003-2005 season...2003-2013 season.

list=['03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15']

for i in range(len(list)):

    with open("try"+list[i]+".csv", "rb") as fp_in, open("Stats"+list[i]+".csv", "wb") as fp_out:
        reader = csv.reader(fp_in)
        writer = csv.writer(fp_out)
        r=[]
        for row in reader:
            r.append(row)

        for i in range(0,21):
            for j in range(21,len(r)):
                if str(r[i][1])==str(r[j][0]):

                    for k in range(1,len(r[j])):
                        if r[j][k]!="":
                            r[i].append(r[j][k])

                
                
            wr = csv.writer(fp_out, dialect='excel')
            wr.writerow(r[i])

            
