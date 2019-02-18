import socket
import time
import os
import ConfigParser

def hash_value(filename):
    asciitot = sum([ord(c) for c in filename])
    return ((asciitot%8)+1)
    
def get_idx_server(hashval):
    configParser = ConfigParser.RawConfigParser()   
    configFilePath = r'/home/ubuntu/config.cfg'
    configParser.read(configFilePath)
    port = int(configParser.get('ports', str(hashval)))
    return (port)

def get_peer_port(peer):
    configParser = ConfigParser.RawConfigParser()   
    configFilePath = r'/home/ubuntu/config.cfg'
    configParser.read(configFilePath)
    port = int(configParser.get('ports', peer))
    return (port)

x = time.time()

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
qry = 's6.py'
port = get_peer_port('peer6')
s.connect(('localhost',port))
s.send('peer6'+' peer4 ' +qry)
resp = s.recv(1024)
print resp

ti = time.time() - x
print ti
