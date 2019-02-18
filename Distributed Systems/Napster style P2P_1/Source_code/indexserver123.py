import socket
import threading
#Creation of multiple threads
class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
	self.lock = threading.Lock()
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
#Creation of the listen Function
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
             statement = client.recv(size)
             split_stat = statement.split()
#Implementation of Register               
	     if(split_stat[0] == 'Add'):
              lst = statement.split()
              val = dict_index.get(lst[1],'NULL')
              if val == 'NULL':
               dict_index.update({lst[1]:lst[2]})
	      else:
	       self.lock.acquire()
	       try:
  	        if lst[2] not in val: 
                 dict_index[lst[1]] = val + ','+lst[2]
 	        else:
	         client.send('File is already present') 
               finally:
	        self.lock.release()
	      client.send('Added '+lst[1]+' to '+lst[2])
#Implementation of Search                   
             if(split_stat[0] == 'Search'):
              lsts = statement.split()
              search_term = lsts[1]
              res = dict_index.get(search_term,'NULL')
	      if res == 'NULL':
	       client.send('Not Presentt')
	      else:
	       client.send(res)
            except:
             client.close()
             return False

if __name__ == "__main__":
    dict_index = {}
    while True:
        try:
            port_num = 6001
            break
        except ValueError:
            pass

    ThreadedServer('',port_num).listen()
