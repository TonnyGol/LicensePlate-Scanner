#-----------------------------------------------------------(All Imports)
import socket
from database import *

DB_PATH = r"Police_DB.db"
LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 8900
#-----------------------------------------------------------(All Tcp network functions)
def create_socket():
#( Creates a Tcp connection so clients could connect to the server )
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (LISTEN_IP, LISTEN_PORT)
        sock.bind(server_address)
    except:
        "An Error Happened"
        return "CONNECTION ERROR"
    else:
        return sock

def get_msg(client_soc):
#( server receives msg from a client and his ip address )
    client_msg, client_addr = client_soc.recvfrom(1024 * 75)
    client_msg = client_msg.decode()
       
    return client_msg, client_addr
    
def send_msg(sock, msg, client_addr):
#( server sends an message to the client back )
    msg_build = msg
    sock.sendto(msg_build.encode(), client_addr)
       
def close_socket(soc):
#( closes the sock so no one could connect now and the server stops )
    soc.close()

def main():
#-----------------------------------------------------------(The server / Main)

    sock = create_socket()
    Sql = SqlObject(DB_PATH)

    while True:
        client_msg, client_addr = get_msg(sock)
        client_msg = eval(client_msg)
        print (client_msg)
        if "Code" in client_msg:
            if client_msg["Code"] == 100:
                msgSend = Sql.checkAllFromCopsTable(client_msg)
            if client_msg["Code"] == 200:
                msgSend = Sql.checkNumInCarsDB(client_msg)
            send_msg(sock, msgSend, client_addr)
    close_socket(sock)
    
if __name__ == "__main__": 
    main()
