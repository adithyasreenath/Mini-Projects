import dota2api
import json
import sys
import time
"""
GET INDIVIDUAL PLAYER MATCH DATA
"""
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")
#acc_id=76561198075407166
#acc_id=376886463
#acc_id=391186900  
#acc_id=108994455
#acc_id=115141438    
acc_id=206244918                  #CHANGE ACCOUNT ID FOR DIFFERENT PLAYERS
folder_name = 'player5'            #CHANGE FOLDER NAME FOR DIFFERENT PLAYERS
#Getting a list of 100 recent matches of player
match_ids_list = []
hist = api.get_match_history(account_id=acc_id,matches_requested=100)
for matches in hist['matches']:
    match_ids_list.append(matches['match_id'])

#Iterating through the match ids to get match details    
for i in match_ids_list:
    time.sleep(1)
    match = api.get_match_details(match_id=i)
    match_path = "matches/" + folder_name + "/" + str(i) + ".json"    
    with open(match_path, 'w') as outfile:  
        json.dump(match, outfile)

def print_match_details_player(id):
    #Opening a specific match file       
    open_path = "matches/" + folder_name + "/" + str(i) + ".json"         
    with open(open_path) as data_file:
        data_loaded = json.load(data_file)
    print(data_loaded)
    
#print_match_details_player(3692804440)