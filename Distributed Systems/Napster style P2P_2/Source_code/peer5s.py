import socket
import threading
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

#Creation of multi thread
class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
#Creation of the listen function
    def listen(self):
        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            client.settimeout(60)
            threading.Thread(target = self.listenToClient,args = (client,address)).start()
#Creation of the listentoclient function
    def listenToClient(self, client, address):
        size = 1024
        while True:
         try:
	  stat = client.recv(size)
	  split_stat = stat.split()
	  path = file_paths.get(split_stat[1])
          from_path = file_paths.get(split_stat[0])
	  cmd ='sudo cp '+ from_path+'/'+split_stat[2]+' '+path
	  val = os.system(cmd)
	  if val == 0:
                client.send('File has been downloaded to :'+path)
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	        hashval = hash_value(split_data[2])
      	        port = get_idx_server(hashval)
	        s.connect(('localhost',port))
                s.send('Add '+split_stat[2]+' '+split_stat[1])
                resp = s.recv(1024)
                s.close()
	  else:
                client.send('Download Failed.')
         except:
               client.close()
               return False

if __name__ == "__main__":
	file_paths = {'peer1':'/home/ubuntu/peer1','peer2':'/home/ubuntu/peer2','peer3':'/home/ubuntu/peer3','peer4':'/home/ubuntu/peer4','peer5':'/home/ubuntu/peer5','peer6':'/home/ubuntu/peer6','peer7':'/home/ubuntu/peer7','peer8':'/home/ubuntu/peer8'}
	while True:
		try:
			port_num = get_peer_port('peer5')
			break
		except ValueError:
			pass

	ThreadedServer('',port_num).listen()
