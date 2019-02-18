import os

#create folders for each peer
os.system('mkdir /home/ubuntu/peer1')
os.system('mkdir /home/ubuntu/peer2')
os.system('mkdir /home/ubuntu/peer3')


#start server processes in the background

os.system('python indexserver.py &')
os.system('python peer1s.py &')
os.system('python peer2s.py &')
os.system('python peer3s.py &')