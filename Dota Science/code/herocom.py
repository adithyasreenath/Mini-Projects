import dota2api
import json
import sys
import time
import os
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms import community
from networkx import edge_betweenness_centrality as betweenness
from operator import itemgetter
from random import random
import itertools

"""
DISPLAY EDGE WEIGHTED GRAPHS FOR A SINGLE PLAYER BASED ON HERO - PAIRS
"""
api = dota2api.Initialise("0CB85F337023047C2BDE72B02DB06754")

hellblazer_id=115141438
puppey_id = 391186900
player2_id = 376886463
arr_match_ids = []
hero_list_count = {}
hero_list = {}
hero_node = []
cmap = []
edge_color_map = []
edge_list1 = []
edge_list2 = []
edge_list3 = []
edge_list4 = []
x = api.get_heroes()
#print(x['heroes'][0])
for i in x['heroes']:
    hero_list_count[i['localized_name']] = 0
    hero_list[i['id']] = i['localized_name']
    
g = nx.Graph()
#print(hero_list.keys())
for x in hero_list.keys():
    hero_node.append(x)
#DISPLAY HERO GRAPH WITH EDFE WEIGHT = 1
edges = combinations(hero_node, 2)
g = nx.Graph()
g.add_nodes_from(hero_node)
g.add_edges_from(edges,weight=1)

plt.figure(3,figsize=(10,8))    
nx.draw_networkx(g, width=0.1, edge_color='blue')
plt.show()
#GET MATCH DETAILS FROM JSON FILES
arr_json = [x for x in os.listdir("matches/player1/") if x.endswith(".json")]
for x in arr_json:
    arr_match_ids.append(os.path.splitext(x)[0])
#print(arr_match_ids)    
for id in arr_match_ids:   
    open_path = "matches/player1/" + str(id) + ".json"          
    with open(open_path) as data_file:
        match_loaded = json.load(data_file)   
    try:
        for player in match_loaded['players']:
            if(player['account_id'] == puppey_id):
                hellid = player['hero_id']
                #print(hellid)
                break
        for player in match_loaded['players']:
            if(player['account_id'] != puppey_id):
                #print(player['hero_id'])
                g[player['hero_id']][hellid]['weight'] += 1                
    except:
        pass
#GROUP NODES BASED ON EDGE WEIGHTS
weight_val = nx.get_edge_attributes(g,'weight')
for key, val in weight_val.items():
    edge_color_map.append(val)
    if(val >= 7):
        edge_list1.append(key)
    elif(val < 7 and val >= 5):
        edge_list2.append(key)
    elif(val < 5 and val >= 3):
        edge_list3.append(key)
    elif(val < 3 and val >= 0):
        edge_list4.append(key)
#print("EDGE WEIGHT",edge_color_map)
#sys.exit()
#for key, value in sorted(weight_val.items(), key=lambda item: (item[1], item[0]), reverse=True):
#    print(key, value)

plt.figure(1, figsize=(10,8))   
plt.title('Greater than 7 games')
nx.draw_networkx(g, width=3, edgelist = edge_list1, edge_color='blue')
plt.savefig("EdgeList1.jpg")
plt.figure(2, figsize=(10,8))    
plt.title('Between 5 and 7 games')
nx.draw_networkx(g, width=3, edgelist = edge_list2, edge_color='blue')
plt.savefig("EdgeList2.jpg")
plt.figure(3, figsize=(10,8))    
plt.title('Between 3 and 4 games')
nx.draw_networkx(g, width=1, edgelist = edge_list3, edge_color='blue')
plt.savefig("EdgeList3.jpg")
plt.figure(4, figsize=(10,8)) 
plt.title('Less than 3 games')   
nx.draw_networkx(g, width=0.1, edgelist = edge_list4, edge_color='blue')
plt.savefig("EdgeList4.jpg")

"""
PARTITION BASED ON GIRVAN NEWMAN ALGORITHM IMPLEMENTED BELOW:
    USEAGE OF THIS ALGORITHM REQUIRES LARGER DATASETS OF MATCHES
    THE RESULTANT OF 100 GAMES PARTITIONED NODES INTO SINGLE-NODE COMMUNITIES
    MATCHES UPWARDS OF 10000 GAMES CAN RESULT IN COMMUNITIES OF 2-3 HERO PAIRS
"""
#def heaviest(G):
#    u, v, w = min(G.edges(data='weight'), key=itemgetter(2))
#    return (u, v)
#
#def most_central_edge(G):
#    centrality = betweenness(G, weight='weight')
#    return min(centrality, key=centrality.get)
#
#k=7
#comp = community.girvan_newman(g, most_valuable_edge=heaviest)
#limited = itertools.takewhile(lambda c: len(c) <= k, comp)
#for communities in limited:
#    print(tuple(sorted(c) for c in communities))
#
print("end")