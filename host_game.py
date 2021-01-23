
import socket
import http.server
import socketserver

DIRECTORY = "./game/"
PORT = 8000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

class MyTCPServer(socketserver.TCPServer):
    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
    
def launch_game(port=PORT):
    
    with MyTCPServer(("", port), Handler) as httpd:
        print(f"Serving at game directory at localhost:{port}")
        httpd.serve_forever()
        
if __name__ == '__main__':
    launch_game()