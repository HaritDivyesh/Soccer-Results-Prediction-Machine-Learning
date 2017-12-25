import csv
import numpy as np

#Code to convert csv to npy

team={'Arsenal':501, 'Aston Villa':502, 'Birmingham City':503, 'Blackburn Rovers':504,'Blackpool':505,'Bolton Wanderers':506,'Burnley':507,'Charlton Athletic':508,'Chelsea':509,'Crystal Palace':510,'Derby County':511,'Everton':512,'Fulham':513,'Hull City':514,'Leeds United':515,'Leicester City': 516,'Liverpool':517,'Manchester City':518,'Manchester United':519,'Middlesbrough':520,'Newcastle United':521,'Norwich City':522,'Portsmouth':523,'Queens Park Rangers':524,'Reading':525,'Sheffield United':526,'Southampton':527,'Stoke City':528,'Sunderland':529,'Swansea City':530,'Tottenham Hotspur':531,'Watford':532,'West Bromwich Albion':533,'West Ham United':534,'Wigan Athletic':535,'Wolverhampton Wanderers':536}

train_list=["03","03-04","03-05","03-06","03-07","03-08","03-09","03-10","03-11","03-12","03-13"]
test_list=["Stats03-04","Stats04-05","Stats05-06","Stats06-07","Stats07-08","Stats08-09","Stats09-10","Stats10-11","Stats11-12","Stats12-13","Stats13-14","Stats 14-15"]

for n in range(len(train_list)):
	with open(train_list[n]+".csv", "rb") as fp_in:
	    reader = csv.reader(fp_in)
	    
	    r=[]
	    for row in reader:
	    	if row[0]!='Team':
	        	r.append(row)

	    for i in range(0,len(r)):
	    	for k in range(85,89):
	    		r[i][k] = str(r[i][k]).split(" (")[0] 
	    	if str(r[i][0]) in team:
				r[i][0] = team[str(r[i][0])]
			for l in range(len(r)):
				r[i][l] = float(r[i][k])

	train = np.array(r)
	np.save('numpyFiles2/train'+train_list[n]+'.npy',train)


for m in range(len(test_list)):
	with open(test_list[m]+".csv", "rb") as fp_in:
	    reader = csv.reader(fp_in)

	    r=[]
	    for row in reader:
	    	if row[0]!='Team':
	        	r.append(row)

	    for i in range(0,len(r)):
	    	for k in range(85,89):
	    		r[i][k] = str(r[i][k]).split(" (")[0]
	    	if str(r[i][0]) in team:
				r[i][0]=team[str(r[i][0])]
 

		                

	test = np.array(r)
	np.save('numpyFiles/test'+test_list[m]+'.npy',test)

