import socket
import ast
import logging
import sys
import os

# Ensure sibling modules (e.g. database.py) are importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager

# Set up logging for the server operations
logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Police_DB.db")
LISTEN_IP = "127.0.0.1"
LISTEN_PORT = 8900


def create_socket():
    """Creates a UDP socket for clients to connect to the server."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (LISTEN_IP, LISTEN_PORT)
        sock.bind(server_address)
        sock.settimeout(1.0) # Add timeout to allow KeyboardInterrupt detection
        logger.info(f"Server running and listening on {LISTEN_IP}:{LISTEN_PORT}")
        return sock
    except Exception as e:
        logger.critical(f"Failed to create socket: {e}")
        return None


def get_msg(client_soc):
    """Receives message from a client and their IP address."""
    try:
        client_msg_bytes, client_addr = client_soc.recvfrom(1024 * 75)
        client_msg = client_msg_bytes.decode('utf-8')
        return client_msg, client_addr
    except socket.timeout:
        return None, None
    except Exception as e:
        logger.error(f"Error receiving message: {e}")
        return None, None


def send_msg(sock, msg, client_addr):
    """Sends a message back to the client."""
    try:
        sock.sendto(msg.encode('utf-8'), client_addr)
        logger.debug(f"Sent response to {client_addr}: {msg}")
    except Exception as e:
        logger.error(f"Error sending message to {client_addr}: {e}")


def main():
    sock = create_socket()
    if not sock:
        return

    db_manager = DatabaseManager(DB_PATH)

    while True:
        try:
            client_msg_raw, client_addr = get_msg(sock)
            if not client_msg_raw:
                continue
                
            logger.info(f"Received raw message from {client_addr}: {client_msg_raw}")

            # Safely parse the incoming string representation of a dictionary
            # Do NOT use eval() as it can execute arbitrary code
            try:
                client_request = ast.literal_eval(client_msg_raw)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Failed to parse incoming message from {client_addr} as Python dict. Error: {e}")
                send_msg(sock, "BAD REQUEST FORMAT", client_addr)
                continue

            if not isinstance(client_request, dict):
                logger.warning(f"Parsed message from {client_addr} is not a dictionary.")
                send_msg(sock, "BAD REQUEST FORMAT", client_addr)
                continue

            msg_send = "UNKNOWN COMMAND"
            if "Code" in client_request:
                command_code = client_request["Code"]
                
                # Code 100: Login Request
                if command_code == 100:
                    logger.info(f"Processing login request (Code 100) from {client_addr}")
                    msg_send = db_manager.check_all_from_cops_table(client_request)
                    
                # Code 200: License Plate Lookup
                elif command_code == 200:
                    logger.info(f"Processing license plate lookup (Code 200) from {client_addr}")
                    msg_send = db_manager.check_num_in_cars_db(client_request)
                else:
                    logger.warning(f"Unknown command code {command_code} from {client_addr}")

            send_msg(sock, msg_send, client_addr)

        except KeyboardInterrupt:
            logger.info("Server shutting down via KeyboardInterrupt.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop during processing: {e}")

    logger.info("Closing server socket.")
    sock.close()


if __name__ == "__main__":
    main()
