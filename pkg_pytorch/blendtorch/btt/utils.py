import socket


def get_primary_ip():
    """Returns the primary IP address of this machine (the one with).

    See https://stackoverflow.com/a/28950776
    Returns the IP address with the default route attached or `127.0.0.1`.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP
