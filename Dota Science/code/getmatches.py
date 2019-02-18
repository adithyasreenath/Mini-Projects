import dota2api
import json
import sys
import time
"""
GET AND STORE 100 RECENT MATCHES
"""
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")


match_ids_list = []
hist = api.get_match_history(matches_requested=100)
for matches in hist['matches']:
    match_ids_list.append(matches['match_id'])

#Iterating through the match ids to get match details    
for i in match_ids_list:
    time.sleep(1)
    match = api.get_match_details(match_id=i)
    match_path = "matches/" + str(i) + ".json"    
    with open(match_path, 'w') as outfile:  
        json.dump(match, outfile)

def print_match_details_player(id):
    #Opening a specific match file       
    open_path = "matches/" + str(id) + ".json"          
    with open(open_path) as data_file:
        data_loaded = json.load(data_file)
    print(data_loaded)

#print_match_details_player(3779575361)
