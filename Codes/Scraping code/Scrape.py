import requests
from bs4 import BeautifulSoup
import numpy as np
import csv

# Code to scrape from StatBunker

lis=[163,172,179,186,200,243,279,323,373,415,449,481]
table=['LeagueTable','AwayLeagueTable','HomeLeagueTable','ClubBookings','TopGoalScorers','Top10KeepersCleanSheets','PlayersGoalScorersDistance','HalfTimeTableWin','HalfTimeTableDraw','HalfTimeTableLose','MostAssists','GoalsPerMatchScored','GoalsPerMatchConceded','GoalsFor','GoalsAgainst','ForPenalty','AgainstPenalty','Top10KeepersCleanSheets',]
url1="https://www.statbunker.com/competitions/"
url2="?comp_id="
year=['03-04','04-05','05-06','06-07','07-08','08-09','09-10','10-11','11-12','12-13','13-14','14-15']

for k in range(len(table)):
	for j in range(len(lis)):
		standings=[]
		url=url1+table[k]+url2+str(lis[j])
	 	page = requests.get(url)
		soup = BeautifulSoup(page.content, 'html.parser')

		with open(table[k]+' '+year[j]+'.csv', 'w') as f:			
			tr = soup.findAll({'tr':True})
			for i in range(len(tr)):
				example=[]
				
				r = tr[i].get_text("|")
				teams = list(r.split("|"))
				players = [team.encode('utf-8') for team in teams]
				if i==0:
					standings.append(players[1:-2:2])
				else:
					print players
					standings.append(players[:-1])
				wr = csv.writer(f, dialect='excel')
    			wr.writerows(standings)
			
