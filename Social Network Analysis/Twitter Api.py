#Imports
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = '<consumer_key>'
consumer_secret = '<consumer_secret>'
access_token = '<access_token>'
access_token_secret = '<access_token_secret>'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    temp = open(filename,'r').read().split('\n')
#    print("names",temp)
    return temp
    pass

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
#    print(help(twitter))
    request = twitter.request('users/lookup', {'screen_name': screen_names})
#    print(type(request))
#    print([u['screen_name'] for u in request])
    return request
    pass


def get_friends(twitter, screen_name):
#    print(screen_name)
    request1 = twitter.request('friends/ids', {'screen_name': screen_name, 'count': 5000})
    rjson = request1.json()
#    print(rjson['ids'])
    return(rjson['ids'])
    pass


def add_all_friends(twitter, users):
    for key in users:
        screen_list = key['screen_name']
        add_req = get_friends(twitter, screen_list) 
        key['friends'] = add_req
#        print(key)
    pass


def print_num_friends(users):
    numdict = {}
    for key in users:
        friends = key['friends']
        screen_list = key['screen_name']
        count = len(friends)
        numdict[screen_list] = count
    for key in sorted(numdict.keys()):
        print(key, numdict[key])
    pass


def count_friends(users):
    c = Counter()
    for key in users:
        friends = key['friends']
        c.update(friends)
#    print(c.most_common(10))
    return c
    pass


def friend_overlap(users):
    users1 = users.copy()
    l = []
    for key1 in users1:
        friends1 = key1['friends']
        screen_list1 = key1['screen_name']
        for key2 in users1:
            if(key1 != key2):        
                screen_list2 = key2['screen_name']
                friends2 = key2['friends']
                count = 0
                for tup in friends1:
                    if tup in friends2:
                        count = count + 1
                l.append((screen_list1, screen_list2, count))
        users1.remove(key1)
    m = sorted(l, key=lambda x:(-x[2], x[0], x[1]))
    return m
    pass


def followed_by_hillary_and_donald(users, twitter):
    friends_hil= []
    friends_donald = []
    friend_id = []
    for key1 in users:
        if(key1['screen_name'] == 'realDonaldTrump'):
            friends_donald = key1['friends']
#            print(friends_donald)
        if(key1['screen_name'] == 'HillaryClinton'):
            friends_hil = key1['friends']
#            print(friends_hil)
    for tup in friends_donald:
        if tup in friends_hil:
            friend_id = tup
#    print(friend_id)
    request = twitter.request('users/lookup', {'user_id': friend_id})  
    fol_name = [u['screen_name'] for u in request]
    return fol_name
    pass


def create_graph(users, friend_counts):
    G = nx.Graph()
    for key1 in users:
        G.add_node(key1['screen_name'])
        friends1 = key1['friends']
        screen_list1 = key1['screen_name']
        for x in friends1:
            G.add_edge(screen_list1, x)
    list_counter = friend_counts.items()
#    print(list_counter)
    for y in list_counter:
        if(y[1] < 2):
            G.remove_node(y[0])
    return G
    pass


def draw_network(graph, users, filename):
    k = []
    for key1 in users:
        screen_name = key1['screen_name']
        k.append(screen_name)
    dict_node = {}
    for i in k:
        dict_node[i] = i
    plt.figure(3,figsize=(15,15)) 
    nx.draw_networkx(graph, pos=nx.spring_layout(graph), labels=dict_node, with_labels=True, width=0.3, edge_color='blue', node_color='red')
    plt.savefig(filename)
    plt.show()
    pass


def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    get_users(twitter, screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()
