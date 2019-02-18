import dota2api
import json
import sys
import time
import os
"""
GET THE LIST OF HERO DETAILS OF ALL HEROES PRESENT IN THE GAME (COUNT = 120 when created)
"""
hellblazer_id=115141438
arr_match_ids = []
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")
match_loaded = []
hero_list = {}
x = api.get_heroes()
#print(x['heroes'][0])
for i in x['heroes']:
    hero_list[i['localized_name']] = 0    
#print(hero_list.keys())


arr_json = [x for x in os.listdir("matches/") if x.endswith(".json")]
try:
    for x in arr_json:
        arr_match_ids.append(os.path.splitext(x)[0])
        for i in arr_match_ids:
            time.sleep(1)
            match = api.get_match_details(match_id=i)
    #        print(match['match_id'])
            try:
                for player_list in match['players']:
    #                print(player_list)
    #                print(player_list['hero_name'])
                    hero_name = player_list['hero_name']
                    hero_list[hero_name] = hero_list[hero_name] + 1
            except:
                pass
except:
    pass
#print(arr_match_ids)    
#for id in arr_match_ids:   
#    open_path = "matches/" + str(id) + ".json"          
#    with open(open_path) as data_file:
#        match_loaded = json.load(data_file)
#    
#print(match_loaded)

    
print(hero_list)