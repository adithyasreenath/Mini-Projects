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

x = time.time()
for i in range(1,10000):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	qry = 'c50.png'
	hashval = hash_value(qry)
	port = get_idx_server(hashval)
	s.connect(('localhost',port))
	s.send('Search '+qry)
	resp = s.recv(1024)
	print resp

ti = time.time() - x
print ti
