import dota2api
import json
import sys
import time
import os
import numpy as np
import matplotlib.pyplot as plt
"""
PLOT BAR GRAPHS FOR 5 PLAYERS BASED ON KDA AND GPM/XPM
"""
bar_win = []
bar_loss = []
bar_kill_win = []
bar_kill_loss = []
bar_death_win = []
bar_death_loss = []
bar_gpm = []
bar_xpm = []
bar_gpm_win = []
bar_xpm_win = []
hellblazer_id=115141438
puppey_id = 391186900
#INITIALIZATION
player_id = [391186900,376886463,108994455,115141438,206244918]
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")
for i in range(1,6):
    arr_json = []
    arr_match_ids = []
    win_count = 0
    loss_count = 0
    kill_count_win = 0
    death_count_win = 0
    assist_count_win = 0
    total_kill_count = 0
    total_death_count = 0
    total_assist_count = 0
    kill_count_loss = 0
    death_count_loss = 0
    assist_count_loss = 0
    average_gold_per_min = 0
    average_xp_per_min = 0
    average_gold_per_min_win = 0
    average_xp_per_min_win = 0
    path1 = "matches/player" + str(i) + "/" 
    arr_json = [x for x in os.listdir(path1) if x.endswith(".json")]
#GETTING MATCH DATA
    for x in arr_json:
        arr_match_ids.append(os.path.splitext(x)[0])
    #print(arr_match_ids)    
    for id in arr_match_ids:   
        open_path = str(path1) + str(id) + ".json"          
        with open(open_path) as data_file:
            match_loaded = json.load(data_file)
        #print(match_loaded)
        for player in match_loaded['players']:
            #print(player['account_id'])
            if(player['account_id'] == player_id[i-1]):
#THE SLOT DATA FOR THE PLAYERS IS PRESENT IN 8-BIT FORMAT. CHECK DOCUMENTATION OF THE WEB-API.
                p_slot = player['player_slot']
                p_slot_bin = int(bin(p_slot)[2:])
                p_slot_bin = str(p_slot_bin)
                #print(p_slot_bin)
#CALCULATIONS OF KDA AND GPM/XPM DATA
                total_kill_count = total_kill_count + player['kills']
                total_death_count = total_death_count + player['deaths']
                total_assist_count = total_assist_count + player['assists']
                average_gold_per_min = average_gold_per_min + player['gold_per_min']
                average_xp_per_min = average_xp_per_min + player['xp_per_min']
                if((len(p_slot_bin) < 8) and (match_loaded['radiant_win'] == True)): #radwin
                    win_count = win_count + 1
                    kill_count_win = kill_count_win + player['kills']
                    death_count_win = death_count_win + player['deaths']
                    assist_count_win = assist_count_win + player['assists']
                    average_gold_per_min_win = average_gold_per_min_win + player['gold_per_min']
                    average_xp_per_min_win = average_xp_per_min_win + player['xp_per_min']
    
                if((len(p_slot_bin) == 8) and (match_loaded['radiant_win'] == False)): #direwin
                    win_count = win_count + 1
                    kill_count_win = kill_count_win + player['kills']
                    death_count_win = death_count_win + player['deaths']
                    assist_count_win = assist_count_win + player['assists']
                    average_gold_per_min_win = average_gold_per_min_win + player['gold_per_min']
                    average_xp_per_min_win = average_xp_per_min_win + player['xp_per_min']
    
    loss_count = 100 - win_count
    kill_count_loss = total_kill_count - kill_count_win
    death_count_loss = total_death_count - death_count_win
    assist_count_loss = total_assist_count - assist_count_win   
    print("PLAYER : ", i)
    print("win_count = ",win_count)
    print("loss_count = ",loss_count)
    print()
    print("total_kill_count = ",total_kill_count)
    print("total_death_count = ",total_death_count)
    print("total_assist_count = ",total_assist_count)
    print()
    print("kill_count_win = ",kill_count_win)
    print("death_count_win = ",death_count_win)
    print("assist_count_win = ",assist_count_win)
    print()
    print("kill_count_loss = ",kill_count_loss)
    print("death_count_loss = ",death_count_loss)
    print("assist_count_loss = ",assist_count_loss)
    print()
    win_per = (win_count/(win_count+loss_count))*100
    kill_perc = (kill_count_win/total_kill_count)*100
    death_perc = (death_count_win/total_death_count)*100
    assist_perc = (assist_count_win/total_assist_count)*100
    print("win percentage = ",win_per)
    print("kill perc = ",kill_perc)
    print("death perc = ",death_perc)
    print("assist perc = ",assist_perc)
    print()
    print("gpm per win = :",average_gold_per_min_win/win_count)
    print("xpm per win = :",average_xp_per_min_win/win_count)
    print("avg gpm total = ",average_gold_per_min/100)
    print("avg xpm total = ",average_xp_per_min/100)
    print("------------")
    bar_win.append(win_count)
    bar_loss.append(loss_count)
    bar_kill_win.append(kill_count_win)
    bar_kill_loss.append(kill_count_loss)
    bar_death_win.append(death_count_win)
    bar_death_loss.append(death_count_loss)
    bar_gpm.append(average_gold_per_min/100)
    bar_xpm.append(average_xp_per_min/100)
    bar_gpm_win.append(average_gold_per_min_win/win_count)
    bar_xpm_win.append(average_xp_per_min_win/win_count)

#WIN COUNT BAR GRAPH    
bar_x =['Pro-player','player1','player2','player3','player4']
y_pos = np.arange(len(bar_x))
plt.figure(1)
plt.bar(y_pos, bar_win, align='center', alpha=1, color='blue')
plt.xticks(y_pos, bar_x)
plt.ylabel('WIN COUNT')
plt.xlabel('PLAYERS')
plt.title('WIN COUNT PER PLAYER')
plt.savefig('Wins.jpg')
plt.show()
#LOSS COUNT BAR GRAPH    
bar_x =['Pro-player','player1','player2','player3','player4']
y_pos = np.arange(len(bar_x))
plt.figure(2)
plt.bar(y_pos, bar_loss, align='center', alpha=1, color='red')
plt.xticks(y_pos, bar_x)
plt.ylabel('LOSS COUNT')
plt.xlabel('PLAYERS')
plt.title('LOSS COUNT PER PLAYER')
plt.savefig('Loss.jpg')
plt.show()
#KILLS DEATHS PER WIN BAR GRAPH
fig, ax = plt.subplots()
index = np.arange(len(bar_x))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, bar_kill_win, bar_width,
                 alpha=1,
                 color='g',
                 label='Kills')
 
rects2 = plt.bar(index + bar_width, bar_death_win, bar_width,
                 alpha=1,
                 color='r',
                 label='Deaths')
 
plt.xlabel('Players')
plt.ylabel('Win')
plt.title('Kills/Deaths per Win')
plt.xticks(index + bar_width, bar_x)
plt.legend()
plt.savefig('KD_win.jpg')
plt.tight_layout()
plt.show()
#KILLS DEATHS PER LOSS BAR GRAPH
fig, ax = plt.subplots()
index = np.arange(len(bar_x))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, bar_death_win, bar_width,
                 alpha=1,
                 color='green',
                 label='Kills')
 
rects2 = plt.bar(index + bar_width, bar_death_loss, bar_width,
                 alpha=1,
                 color='red',
                 label='Deaths')
 
plt.xlabel('Players')
plt.ylabel('Loss')
plt.title('Kills/Deaths per Loss')
plt.xticks(index + bar_width, bar_x)
plt.legend()
 
plt.tight_layout()
plt.savefig('KD_loss.jpg')
plt.show()
#GPM XPM BAR GRAPH
fig, ax = plt.subplots()
index = np.arange(len(bar_x))
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, bar_gpm_win, bar_width,
                 alpha=1,
                 color='yellow',
                 label='GPM')
 
rects2 = plt.bar(index + bar_width, bar_xpm_win, bar_width,
                 alpha=1,
                 color='grey',
                 label='XPM')
 
plt.xlabel('Players')
plt.ylabel('Minutes')
plt.title('Gold per minute/Experience per minute per Win')
plt.xticks(index + bar_width, bar_x)
plt.legend()
 
plt.tight_layout()
plt.savefig('GPM_XPM.jpg')
plt.show()