# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 21:49:44 2018

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Date: 28 Apr 2018
Author: Chris Guise / Arizona State University
Python for Engineers Final Project - Spring 2018

*******************************************************************************
*                                                                             *
*                            ImportData.py:                                   *
*      Import Retrosheet data with minor preprocessing, then                  *
*      Calculate advanced stats:                                              *
*      Elo rating throughout each season, pythagorean win/loss record,        *
*      month-by-month stats,                                                  *
*      (win/loss record, home/away split, R/H/E by month, run differential,   *
*      record in games decided by one run or less)                            *
*                                                                             *
*******************************************************************************

Overview:
1) Import the csv file
2) Create seed data for dataframe
3) Crunch dataset. This takes 5-10 minutes.
4) Output RetrosheetCrunched.csv
    
"""

#need numpy functions
import numpy as np
from io import StringIO

#need pandas for file reading
import pandas as pd

#import the Retrosheet dataset
df = pd.read_csv('./Retrosheet_Games1952_2015sorted.csv', encoding='latin-1')

# ------------------------------------------------------------
#
# Preprocessing needed:
# 1) GAME_DATE column in format YYYYMMDD      i.e. 19520415    
# 2) Sort by GAME_DATE                        
# 3) MONTH column (extracted from GAME_DATE), i.e. 4/5/6/7/8/9/10
# 4) HOME column (home team ID) i.e. 1952_WS1
# 5) AWAY column (away team ID) i.e. 1952_BRO
# 6) WINN_TEAM column (used if statement in Excel)
# 7) LOSE_TEAM column (used if statement in Excel)
#
# ------------------------------------------------------------

#Return the column headers names
data_columns = list(df)

#Create dataframe with seed data
df2 = pd.DataFrame(0, index = ['2001_ARI'], columns = ['3W',  '3L',  '3_1RUN_W'  ,'3_1RUN_L'  ,  '3_1RUN_pct' , '3PCT', '3H', '3R', '3RA', '3RD', '3E', 
                                                       '4W',  '4L',  '4_1RUN_W'  ,'4_1RUN_L'  ,  '4_1RUN_pct' , '4PCT', '4H', '4R', '4RA', '4RD', '4E',
                                                       '5W',  '5L',  '5_1RUN_W'  ,'5_1RUN_L'  ,  '5_1RUN_pct' , '5PCT', '5H', '5R', '5RA', '5RD', '5E',
                                                       '6W',  '6L',  '6_1RUN_W'  ,'6_1RUN_L'  ,  '6_1RUN_pct' , '6PCT', '6H', '6R', '6RA', '6RD', '6E',
                                                       '7W',  '7L',  '7_1RUN_W'  ,'7_1RUN_L'  ,  '7_1RUN_pct' , '7PCT', '7H', '7R', '7RA', '7RD', '7E', 
                                                       '8W',  '8L',  '8_1RUN_W'  ,'8_1RUN_L'  ,  '8_1RUN_pct' , '8PCT', '8H', '8R', '8RA', '8RD', '8E',
                                                       '9W',  '9L',  '9_1RUN_W'  ,'9_1RUN_L'  ,  '9_1RUN_pct' , '9PCT', '9H', '9R', '9RA', '9RD', '9E', 
                                                       '10W', '10L', '10_1RUN_W' ,'10_1RUN_L'  , '10_1RUN_pct', '10PCT', '10H', '10R', '10RA', '10RD', '10E',
                                                       'homeW','homeL','homePct','awayW','awayL', 'awayPct', 'totalW' , 'totalL',
                                                       'totalPct','totalR', 'totalRA', 'totalRD', 
                                                       'total_1RUN_W', 'total_1RUN_L','total_1RUN_pct','pythagPct', 'pythagDiff', 'eloRating'])
#Set 2001 Dbacks to ELO=1500 initially, as the "seed" dataframe
df2.loc['2001_ARI', 'eloRating'] = 1500

#Set Elo factor scaling.
Kf = 15

#Parse the Retrosheet array into a format by month, then with a season-long home/away record and Elo calculation
#4W, 4L, 4WLPCT, 4_1RUN, 4H, 4R, 4RA, 4RD, 4H, 4E (for each month 4,5,6,7,8,9. Playoffs are in October.)

for i in range(0, len(df)-1):
    
    #These are hard coded based off the sample file. 
    #In future can use column labels to make more user friendly 
    rowData = df.loc[i].tolist()
    hometeam = rowData[11]
    awayteam = rowData[12]
    winnteam = rowData[25]
    loseteam = rowData[26]
    month    = rowData[5]
    homehits = rowData[20]
    awayhits = rowData[19]
    homeruns = rowData[18]
    awayruns = rowData[17]
    homeerrs = rowData[22]
    awayerrs = rowData[21]
    
    #Column names - so data goes to the right place
    wMonthCol   = str(month) + 'W'
    lMonthCol   = str(month) + 'L'
    wlPctCol    = str(month) + 'PCT'
    hitMonthCol = str(month) + 'H'
    rMonthCol   = str(month) + 'R'
    raMonthCol  = str(month) + 'RA'
    rdMonthCol  = str(month) + 'RD'
    errMonthCol = str(month) + 'E'
    month_oneRunWCol   = str(month) + '_1RUN_W'
    month_oneRunLCol   = str(month) + '_1RUN_L'
    month_oneRunPctCol     = str(month) + '_1RUN_pct'
    
    if not (hometeam in df2.index.values):
        #df2.loc[hometeam] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1500]
        df2.loc[hometeam,:] = 0
        df2.loc[hometeam,'eloRating'] = 1500
    
    if not (awayteam in df2.index.values):
       # df2.loc[awayteam] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1500]    
        df2.loc[awayteam,:] = 0
        df2.loc[awayteam,'eloRating'] = 1500
        
    #return "current" Elo ratings (before the game)
    homeElo_old = float(df2.loc[hometeam, 'eloRating'])
    awayElo_old = float(df2.loc[awayteam, 'eloRating'])
    
    homeElo = homeElo_old
    awayElo = awayElo_old
        
    #Who won the game? Update the monthly, home, away, total win/loss stats, and Elo rating
    if (winnteam == hometeam):
        
        df2.loc[hometeam, 'totalW'] = df2.loc[hometeam, 'totalW'] + 1
        df2.loc[hometeam, 'homeW'] = df2.loc[hometeam, 'homeW'] + 1
        df2.loc[hometeam, wMonthCol] = df2.loc[hometeam, wMonthCol] + 1
  
        df2.loc[awayteam, 'totalL'] = df2.loc[awayteam, 'totalL'] + 1
        df2.loc[awayteam, 'awayL'] = df2.loc[awayteam, 'awayL'] + 1      
        df2.loc[awayteam, lMonthCol] = df2.loc[awayteam, lMonthCol] + 1

        
        #Update home Elo...home team perspective
        homeExpectedValue = 1 / (1 + (10**(-1 * (homeElo_old - awayElo_old)/400.0)))
        homeElo = homeElo_old + (Kf * (1 - homeExpectedValue))
        
        #Update away Elo...away team perspective
        awayExpectedValue = 1 / (1 + (10**(-1 * (awayElo_old - homeElo_old)/400.0)))
        awayElo = awayElo_old + (Kf * (0 - awayExpectedValue))
        
        #Update Elo ratings
        df2.loc[hometeam, 'eloRating'] = homeElo 
        df2.loc[awayteam, 'eloRating'] = awayElo
        
    if (winnteam == awayteam):

        df2.loc[awayteam, 'totalW'] = df2.loc[awayteam, 'totalW'] + 1
        df2.loc[awayteam, 'awayW'] = df2.loc[awayteam, 'awayW'] + 1
        df2.loc[awayteam, wMonthCol] = df2.loc[awayteam, wMonthCol] + 1
  
        df2.loc[hometeam, 'totalL'] = df2.loc[hometeam, 'totalL'] + 1
        df2.loc[hometeam, 'homeL'] = df2.loc[hometeam, 'homeL'] + 1      
        df2.loc[hometeam, lMonthCol] = df2.loc[hometeam, lMonthCol] + 1        
        
        #Update home Elo...home team perspective
        homeExpectedValue = 1 / (1 + (10**(-1 * (homeElo_old - awayElo_old)/400.0)))
        homeElo = homeElo_old + (Kf * (0 - homeExpectedValue))
        
        #Update away Elo...away team perspective
        awayExpectedValue = 1 / (1 + (10**(-1 * (awayElo_old - homeElo_old)/400.0)))
        awayElo = awayElo_old + (Kf * (1 - awayExpectedValue))
        
        #Update Elo ratings
        df2.loc[hometeam, 'eloRating'] = homeElo 
        df2.loc[awayteam, 'eloRating'] = awayElo

    
    #update WLPct results (Home)
    #MonthPct
    df2.loc[hometeam, wlPctCol] = df2.loc[hometeam, wMonthCol] / (df2.loc[hometeam, wMonthCol] + df2.loc[hometeam, lMonthCol])
    #HomePct
    df2.loc[hometeam, 'homePct'] = df2.loc[hometeam, 'homeW'] / (df2.loc[hometeam, 'homeW'] + df2.loc[hometeam, 'homeL'])
    #AwayPct
    #df2.loc[hometeam, 'awayPct'] = df2.loc[hometeam, 'awayW'] / (df2.loc[hometeam, 'awayW'] + df2.loc[hometeam, 'awayL'])    
    #TotalPct
    df2.loc[hometeam, 'totalPct'] = df2.loc[hometeam, 'totalW'] / (df2.loc[hometeam, 'totalW'] + df2.loc[hometeam, 'totalL'])   
    
    #update WLPct results (Away)
    #MonthPct
    df2.loc[awayteam, wlPctCol] = df2.loc[awayteam, wMonthCol] / (df2.loc[awayteam, wMonthCol] + df2.loc[awayteam, lMonthCol])
    #HomePct
    #df2.loc[awayteam, 'homePct'] = df2.loc[awayteam, 'homeW'] / (df2.loc[awayteam, 'homeW'] + df2.loc[awayteam, 'homeL'])
    #AwayPct
    df2.loc[awayteam, 'awayPct'] = df2.loc[awayteam, 'awayW'] / (df2.loc[awayteam, 'awayW'] + df2.loc[awayteam, 'awayL'])    
    #TotalPct
    df2.loc[awayteam, 'totalPct'] = df2.loc[awayteam, 'totalW'] / (df2.loc[awayteam, 'totalW'] + df2.loc[awayteam, 'totalL'])   

     #1-run games:
    if(np.abs(homeruns - awayruns) == 1):
        
        #Increment win and loss counts
        if (winnteam == hometeam):
            df2.loc[hometeam,month_oneRunWCol] = df2.loc[hometeam,month_oneRunWCol] + 1
            df2.loc[awayteam,month_oneRunLCol] = df2.loc[awayteam,month_oneRunLCol] + 1
            df2.loc[hometeam,'total_1RUN_W'] = df2.loc[hometeam,'total_1RUN_W'] + 1
            df2.loc[awayteam,'total_1RUN_L'] = df2.loc[awayteam,'total_1RUN_L'] + 1

        if (winnteam == awayteam):
            df2.loc[hometeam,month_oneRunWCol] = df2.loc[hometeam,month_oneRunLCol] + 1
            df2.loc[awayteam,month_oneRunLCol] = df2.loc[awayteam,month_oneRunWCol] + 1
            df2.loc[hometeam,'total_1RUN_L'] = df2.loc[hometeam,'total_1RUN_L'] + 1
            df2.loc[awayteam,'total_1RUN_W'] = df2.loc[awayteam,'total_1RUN_W'] + 1
        
        df2.loc[hometeam,month_oneRunPctCol] = df2.loc[hometeam,month_oneRunWCol] / (df2.loc[hometeam,month_oneRunWCol] + df2.loc[hometeam,month_oneRunLCol])
        df2.loc[awayteam,month_oneRunPctCol] = df2.loc[awayteam,month_oneRunWCol] / (df2.loc[awayteam,month_oneRunWCol] + df2.loc[awayteam,month_oneRunLCol])

        df2.loc[hometeam,'total_1RUN_pct'] = df2.loc[hometeam,'total_1RUN_W'] / (df2.loc[hometeam,'total_1RUN_W'] + df2.loc[hometeam,'total_1RUN_L'])
        df2.loc[awayteam,'total_1RUN_pct'] = df2.loc[awayteam,'total_1RUN_W'] / (df2.loc[awayteam,'total_1RUN_W'] + df2.loc[awayteam,'total_1RUN_L']) 
    
            

    #Update Pythag Win Percents       
    df2.loc[hometeam, 'totalR']  =  df2.loc[hometeam, 'totalR']  + homeruns
    df2.loc[hometeam, 'totalRA'] =  df2.loc[hometeam, 'totalRA'] + awayruns
    df2.loc[hometeam, 'totalRD'] =  df2.loc[hometeam, 'totalR'] - df2.loc[hometeam, 'totalRA']
    
    df2.loc[awayteam, 'totalR']  =  df2.loc[awayteam, 'totalR']  + awayruns
    df2.loc[awayteam, 'totalRA'] =  df2.loc[awayteam, 'totalRA'] + homeruns
    df2.loc[awayteam, 'totalRD'] =  df2.loc[awayteam, 'totalR'] - df2.loc[awayteam, 'totalRA']
    
    if(df2.loc[hometeam, 'totalRD'] != 0):
        df2.loc[hometeam, 'pythagPct']   = df2.loc[hometeam, 'totalR']**1.83 / (df2.loc[hometeam, 'totalR']**1.83 + df2.loc[hometeam, 'totalRA']**1.83)
        df2.loc[hometeam, 'pythagDiff']  = df2.loc[hometeam, 'totalPct'] - df2.loc[hometeam, 'pythagPct']

    if(df2.loc[awayteam, 'totalRD'] != 0):    
        df2.loc[awayteam, 'pythagPct']   = df2.loc[awayteam, 'totalR']**1.83 / (df2.loc[awayteam, 'totalR']**1.83 + df2.loc[awayteam, 'totalRA']**1.83)
        df2.loc[awayteam, 'pythagDiff']  = df2.loc[awayteam, 'totalPct'] - df2.loc[awayteam, 'pythagPct'] 

    
    #Update hits, runs scored, runs allowed, errors (Home)
    df2.loc[hometeam, hitMonthCol] = df2.loc[hometeam, hitMonthCol] + homehits
    df2.loc[hometeam, rMonthCol] = df2.loc[hometeam, rMonthCol] + homeruns
    df2.loc[hometeam, raMonthCol] = df2.loc[hometeam, raMonthCol] + awayruns
    df2.loc[hometeam, errMonthCol] = df2.loc[hometeam, errMonthCol] + homeerrs
    df2.loc[hometeam, rdMonthCol] = df2.loc[hometeam, rdMonthCol] + (homeruns - awayruns)
    
    #Update hits, runs scored, runs allowed, errors (Away)
    df2.loc[awayteam, hitMonthCol] = df2.loc[awayteam, hitMonthCol] + awayhits
    df2.loc[awayteam, rMonthCol] = df2.loc[awayteam, rMonthCol] + awayruns
    df2.loc[awayteam, raMonthCol] = df2.loc[awayteam, raMonthCol] + homeruns
    df2.loc[awayteam, errMonthCol] = df2.loc[awayteam, errMonthCol] + awayerrs
    df2.loc[awayteam, rdMonthCol] = df2.loc[awayteam, rdMonthCol] + (awayruns - homeruns)
    

df2.to_csv('MLBdata.csv')
