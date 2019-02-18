import socket

while(1):
 query = '' 
 inp = raw_input('Welcome Peer2:\n1:Register\n2:Search\n3:Obtain\n')
#Register for peer1
 if inp == '1':
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(('localhost',6001))
  file = raw_input('Enter file_name and peer_name\n')
  query = 'Add '+file
  s.send(query)
  response =  s.recv(1024)
  s.close()
#Search for peer2
 if inp == '2':
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(('localhost',6001))
  file = raw_input('please enter file name to search\n')
  query = 'Search '+file
  print type(query)
  s.send(query)
  response =  s.recv(1024)
  s.close()
  print('Files are present in:', response)
#Obtain for peer2 
 if inp == '3':
  peerdl = raw_input('Enter the peer_name and the file_name\n')
  peerdl_sp = peerdl.split()
  if peerdl_sp[0] == 'peer2':
	print('Invalid Input\n')
  if peerdl_sp[0] == 'peer1':
    s1  = socket.socket( socket.AF_INET, socket.SOCK_STREAM)
    s1.connect(('localhost',6002))
    req = peerdl_sp[0]+' '+'peer2'+' '+peerdl_sp[1]
    s1.send(req)
    response = s1.recv(1024)
    s1.close()
  if peerdl_sp[0] == 'peer3':
    s1  = socket.socket( socket.AF_INET, socket.SOCK_STREAM)
    s1.connect(('localhost',6004))
    req = peerdl_sp[0]+' '+'peer2'+' '+peerdl_sp[1]
    s1.send(req)
    response = s1.recv(1024)
    s1.close()
  print response
