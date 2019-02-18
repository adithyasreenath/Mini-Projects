import dota2api
import json
import sys
import time
import os
"""
GET THE HERO DATA FOR A SINGLE PLAYER AND DISPLAY THE COUNT FOR EACH PLAYED HERO
"""
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")

hellblazer_id=115141438
arr_match_ids = []
hero_list_count = {}
hero_list = {}
x = api.get_heroes()
#print(x['heroes'][0])
for i in x['heroes']:
    hero_list_count[i['localized_name']] = 0
    hero_list[i['id']] = i['localized_name']
#print(hero_list_count)


arr_json = [x for x in os.listdir("matches/Hellblazer/") if x.endswith(".json")]
for x in arr_json:
    arr_match_ids.append(os.path.splitext(x)[0])
#print(arr_match_ids)    
for id in arr_match_ids:   
    open_path = "matches/Hellblazer/" + str(id) + ".json"          
    with open(open_path) as data_file:
        match_loaded = json.load(data_file)   
    try:
        for player in match_loaded['players']:
            hero_name = player['hero_name']
            hero_list_count[hero_name] = hero_list_count[hero_name] + 1
    except:
        pass

print("HERO LIST")
print(hero_list)        
print("HERO COUNTS")
for key in hero_list_count:
    print(key,hero_list_count[key])
