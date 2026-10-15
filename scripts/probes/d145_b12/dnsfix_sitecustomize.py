"""B12 install-time only: hyde06's tailscale resolver (100.100.100.100) fails to resolve public names while the
network itself works. When the system lookup fails, resolve the name with `dig +short @8.8.8.8` and hand the IPv4
address to the original getaddrinfo. TLS keeps verifying against the host name (server_hostname is unchanged).
Used only through PYTHONPATH for conda / pip installs; never on any run's path."""
import socket
import subprocess

_orig = socket.getaddrinfo
_cache = {}


def _dig(host):
    if host not in _cache:
        try:
            out = subprocess.run(["dig", "+short", "@8.8.8.8", host, "A"], capture_output=True, text=True,
                                 timeout=10).stdout.split()
        except Exception:  # noqa: BLE001
            out = []
        _cache[host] = [s for s in out if s.count(".") == 3 and all(p.isdigit() for p in s.split("."))]
    return _cache[host]


def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _orig(host, port, family, type, proto, flags)
    except socket.gaierror:
        if not isinstance(host, str):
            raise
        ips = _dig(host)
        if not ips:
            raise
        res = []
        for ip in ips:
            res.extend(_orig(ip, port, socket.AF_INET, type, proto, flags))
        return res


socket.getaddrinfo = getaddrinfo
