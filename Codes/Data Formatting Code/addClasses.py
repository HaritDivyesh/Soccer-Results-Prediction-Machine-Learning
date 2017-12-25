import csv

#File to assign classes to points. Points 0-10, 10-20, 20-30 -> Class 1,2,3. Points 30-60 in steps of 5 are in classes 4-9. Rest are in steps of 3.

files = ['03','03-04','03-05','03-06','03-07','03-08','03-09','03-10','03-11','03-12','03-13']
for f in range(len(files)):
    with open(files[f]+".csv", "rb") as fp_in:
        reader = csv.reader(fp_in)
        
        r=[]
        for row in reader:
            r.append(row)
        r[0].append('Points_Class')
        for i in range(1,len(r)):
            points = int(r[i][8])
            if points<=30:
                c=(points/10)+1
            if points>30 and points<=60:
                c=(points/5)-2
            if points>61:
                c=(points/3)-10
            
            r[i].append(c)
    fp_in.close()           
            
    with open(files[f]+".csv", "wb") as fp_out:  
        writer = csv.writer(fp_out)  
        for i in range(len(r)):
            wr = csv.writer(fp_out, dialect='excel')
            wr.writerow(r[i])

            
