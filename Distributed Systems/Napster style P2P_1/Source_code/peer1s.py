import socket
import threading
import os
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
		        s.connect(('localhost',6001))
			s.send('Add '+split_stat[2]+' '+split_stat[1])
			resp = s.recv(1024)
			s.close()
		else:
			client.send('Download Failed.')
         except:
               client.close()
               return False

if __name__ == "__main__":
	file_paths = {'peer1':'/home/ubuntu/peer1','peer2':'/home/ubuntu/peer2','peer3':'/home/ubuntu/peer3'}
	while True:
	# port_num = input("Port? ")
		try:
			port_num = 6002
			break
		except ValueError:
			pass

	ThreadedServer('',port_num).listen()
