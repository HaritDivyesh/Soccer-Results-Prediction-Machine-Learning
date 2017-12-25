import requests
from bs4 import BeautifulSoup
import numpy as np
import csv

# More scraping

lis=[163,172,179,186,200,243,279,323,373,415,449,481]
table=['TeamsGoalScorersTypeOfPlay']
url1="https://www.statbunker.com/competitions/"
url2="?comp_id="
year=['03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15']


for k in range(len(table)):
	for j in range(len(lis)):
		url=url1+table[k]+url2+str(lis[j])
	 	page = requests.get(url)
		soup = BeautifulSoup(page.content, 'html.parser')

		with open(table[k]+' '+year[j]+'.csv', 'w') as f:
			f.write('Clubs'+ " " + 'Goals'+ " " + 'Open Play'+ " " + 'Cross'+ " " + 'Free Kick'+ " " + 'Direct Free Kick'+ " " + 'Throw in'+ " " + 'Penalty'+ " " + 'Corner'+ " " + 'Other'+ "\n")
			tr = soup.findAll({'td':True})
			for l in range(20):
				example=[]

				for i in range(11):

					r = tr[(11*l)+i].get_text("|")
					#rows = r[:-5]
					teams = list(r.split("|"))
					attr = [team.encode('utf-8') for team in teams]
					if i==0 and attr==['-']:
						pass
					elif attr!=['More']:
						example.append(''.join(attr))

					wr = csv.writer(f, dialect='excel')
	    			wr.writerow(example)
				

