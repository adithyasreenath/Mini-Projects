import socket
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
    
while(1):
 query = '' 
 inp = raw_input('Welcome peer1:\n1:Register\n2:Search\n3:Obtain\n')
 if inp == '1':
#Registering for peer1
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  file = raw_input('please enter file and peer\n')
  comps = file.split(' ')
  hashval = hash_value(comps[0])
  port = get_idx_server(hashval)
  s.connect(('localhost',port))
  query = 'Add '+file
  s.send(query)
  response =  s.recv(1024)
  s.close()
#Searching for peer1
 if inp == '2':
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  file = raw_input('Please enter file name to search\n')
  hashval = hash_value(file)
  port = get_idx_server(hashval)
  s.connect(('localhost',port))
  query = 'Search '+file
  print type(query)
  s.send(query)
  response =  s.recv(1024)
  s.close()
  print('Files are present in:', response)
#Obtain for peer1
 if inp == '3':
  peerdl = raw_input('Enter the peer_name and the file_name\n')
  peerdl_sp = peerdl.split()
  if peerdl_sp[0] == 'peer1':
	print('Invalid input\n')
  else:
   s1  = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s1.connect(('localhost',get_peer_port(peerdl_sp[0])))
   req = peerdl_sp[0]+' '+'peer1'+' '+peerdl_sp[1]
   s1.send(req)
   response = s1.recv(1024)
   s1.close()
  
  print response
