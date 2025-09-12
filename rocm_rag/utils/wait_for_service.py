import socket
import time

def wait_for_port(host, port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            if result == 0:
                print(f"Port open: {host}:{port}")
                return True
        print(f"Waiting for {host}:{port}...")
        time.sleep(5)
    raise TimeoutError(f"Port not available: {host}:{port}")