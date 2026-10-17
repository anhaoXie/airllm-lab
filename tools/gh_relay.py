#!/usr/bin/env python3
"""
SNI 分流 TLS 中继 / SNI-routing TLS relay  —  127.0.0.1:443

背景
----
Steam++（Watt Toolkit）在 hosts 里把 47 个域名指向 127.0.0.1，指望自己的本地代理在
443 端口接住并解密转发。但它的 Steam++.Accelerator.exe 实测只监听 80 和 444，
443 无人接听 —— 于是这些域名全部指向死端口。

本脚本补上那个缺失的 443 监听：读出 TLS ClientHello 里的 SNI，连到该域名的真实 IP，
把字节原样双向转发。SNI 原样透传，所以证书校验照常通过。

设计要点（都是被实测逼出来的）
------------------------------
1. **运行时自己查 DNS**：不写死 IP。直连 223.5.5.5 / 119.29.29.29 / 8.8.8.8 发
   原始 A 记录查询（绕过被污染的 hosts）。这样 Steam++ 那 47 个域名全都自动可用，
   也不需要一个个去探 IP。
2. **记住赢家**：每成功连上一个 IP 就缓存，下次同一个域名优先用它。
   这解决了首版"按顺序串行试、每个等 6 秒、最坏 42 秒"的灾难。
3. **串行探测超时压到 2.5 秒**：GFW 的 SNI 阻断表现为"TCP 能连上但 TLS 握手被丢弃"，
   所以靠能不能收到 ServerHello 判断该 IP 是否可用。
4. **拒绝回环**：hosts 里未被路由的域名解析后仍是 127.0.0.1，若直连会打到中继自己，
   每层再开一个连接 → 无限递归。
5. **日志限流**：浏览器会疯狂请求 github.githubassets.com 这类资源，不限流会刷爆日志。

不需要管理员权限，不修改任何系统文件，关掉即恢复原状。
"""
import socket
import struct
import random
import threading
import time
import os
import sys

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 443
UPSTREAM_PORT = 443
PROBE_TIMEOUT = 2.5          # 单次"连上并等到 ServerHello"的预算
DNS_TIMEOUT = 3.0
MAX_HELLO = 16384

DNS_SERVERS = ["223.5.5.5", "119.29.29.29", "8.8.8.8"]
DNS_CACHE_TTL = 300          # 秒

# 少数关键域名的优先候选（DNS 结果会追加在后面）。
# 这些是实测在本网络比较抗 SNI 阻断的 GitHub 入口。
PREFERRED = {
    "github.com":          ["140.82.113.3", "140.82.114.3", "140.82.116.3", "140.82.121.3"],
    "gist.github.com":     ["140.82.113.3", "140.82.114.3"],
    "api.github.com":      ["20.205.243.168", "140.82.113.6"],
    "codeload.github.com": ["20.205.243.165"],
}

LOG_PATH = os.environ.get("RELAY_LOG", "")
_plock = threading.Lock()
_counter = [0]
_winner = {}                  # host -> ip   上次成功的 IP
_dns_cache = {}               # host -> (ts, [ip])
_noise = {}                   # 用于日志限流


def log(msg, key=None, every=1):
    """key 不为 None 时按 key 限流：同类消息每 every 条才输出一次。"""
    if key is not None:
        with _plock:
            _noise[key] = _noise.get(key, 0) + 1
            n = _noise[key]
        if n % every != 0:
            return
        msg = "%s  (x%d)" % (msg, n)
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _plock:
        print(line, flush=True)
        if LOG_PATH:
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


# ---------------------------------------------------------------- DNS (raw)

def _encode_name(name):
    out = b""
    for label in name.split("."):
        if label:
            out += bytes([len(label)]) + label.encode("ascii", "ignore")
    return out + b"\x00"


def _skip_name(data, p):
    while p < len(data):
        if data[p] == 0:
            return p + 1
        if data[p] & 0xC0 == 0xC0:      # 压缩指针
            return p + 2
        p += 1 + data[p]
    return p


def dns_a(host):
    """直连公共 DNS 查 A 记录；绕过本机被污染的 hosts 与 DNS 缓存。"""
    qid = random.randint(0, 0xFFFF)
    pkt = (struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0)
           + _encode_name(host) + struct.pack(">HH", 1, 1))
    for srv in DNS_SERVERS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(DNS_TIMEOUT)
            s.sendto(pkt, (srv, 53))
            data, _ = s.recvfrom(4096)
            s.close()
        except Exception:
            continue
        if len(data) < 12:
            continue
        rid, flags, qd, an, _ns, _ar = struct.unpack(">HHHHHH", data[:12])
        if rid != qid or not (flags & 0x8000) or an == 0:
            continue
        p = 12
        for _ in range(qd):
            p = _skip_name(data, p) + 4
        ips = []
        for _ in range(an):
            p = _skip_name(data, p)
            if p + 10 > len(data):
                break
            rtype, _rclass, _ttl, rdlen = struct.unpack(">HHIH", data[p:p + 10])
            p += 10
            rdata = data[p:p + rdlen]
            p += rdlen
            if rtype == 1 and rdlen == 4:
                ips.append(socket.inet_ntoa(rdata))
        if ips:
            return ips
    return []


def candidates(host):
    """候选 IP：上次成功的排最前，其次优先列表，最后 DNS 结果。"""
    out = []
    w = _winner.get(host)
    if w:
        out.append(w)
    for ip in PREFERRED.get(host, []):
        if ip not in out:
            out.append(ip)
    now = time.time()
    cached = _dns_cache.get(host)
    if cached and now - cached[0] < DNS_CACHE_TTL:
        ips = cached[1]
    else:
        ips = dns_a(host)
        _dns_cache[host] = (now, ips)
    for ip in ips:
        if ip not in out:
            out.append(ip)
    return out


# ---------------------------------------------------------------- TLS / relay

def read_client_hello(sock):
    buf = b""
    while len(buf) < MAX_HELLO:
        if len(buf) >= 5:
            rec_len = int.from_bytes(buf[3:5], "big")
            if 5 + rec_len <= len(buf):
                return buf
        try:
            chunk = sock.recv(4096)
        except Exception:
            return buf
        if not chunk:
            return buf
        buf += chunk
    return buf


def parse_sni(data):
    try:
        if len(data) < 5 or data[0] != 0x16:
            return None
        p = 5
        if data[p] != 0x01:
            return None
        p += 4
        p += 2
        p += 32
        p += 1 + data[p]
        cs_len = int.from_bytes(data[p:p + 2], "big")
        p += 2 + cs_len
        p += 1 + data[p]
        ext_total = int.from_bytes(data[p:p + 2], "big")
        p += 2
        end = p + ext_total
        while p + 4 <= end:
            et = int.from_bytes(data[p:p + 2], "big")
            el = int.from_bytes(data[p + 2:p + 4], "big")
            p += 4
            if et == 0x0000:
                q = p + 2 + 1
                nl = int.from_bytes(data[q:q + 2], "big")
                return data[q + 2:q + 2 + nl].decode("ascii", "ignore")
            p += el
    except Exception:
        pass
    return None


def connect_upstream(host, head, ips):
    """依次试候选 IP；靠"能否收到 ServerHello"判断是否被阻断。"""
    last = "no candidate"
    for ip in ips:
        if ip.startswith("127.") or ip in ("::1", "0.0.0.0"):
            last = "loopback"
            continue
        up = None
        try:
            up = socket.create_connection((ip, UPSTREAM_PORT), timeout=PROBE_TIMEOUT)
            up.sendall(head)
            up.settimeout(PROBE_TIMEOUT)
            first = up.recv(65536)          # 期待 TLS ServerHello
            if not first:
                raise IOError("closed before ServerHello")
            up.settimeout(None)
            _winner[host] = ip
            return up, ip, first
        except Exception as e:
            last = "%s" % type(e).__name__
            if up is not None:
                try:
                    up.close()
                except Exception:
                    pass
    raise IOError(last)


def pump(src, dst):
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except Exception:
        pass
    finally:
        for s, how in ((dst, socket.SHUT_WR), (src, socket.SHUT_RD)):
            try:
                s.shutdown(how)
            except Exception:
                pass


def handle(client):
    with _plock:
        _counter[0] += 1
        cid = _counter[0]
    upstream = None
    try:
        client.settimeout(15)
        head = read_client_hello(client)
        if not head:
            return
        sni = parse_sni(head)
        if not sni:
            log("#%d no SNI, dropping %dB" % (cid, len(head)), key="nosni", every=50)
            return
        ips = candidates(sni)
        if not ips:
            log("#%d %s : no IP (DNS failed)" % (cid, sni), key="nodns", every=50)
            return
        # 防自环：全是回环地址说明解析没救回来
        if all(ip.startswith("127.") or ip in ("::1", "0.0.0.0") for ip in ips):
            log("#%d %s -> loopback, refusing" % (cid, sni), key="loop:" + sni, every=200)
            return
        t0 = time.time()
        upstream, ip, first = connect_upstream(sni, head, ips)
        client.sendall(first)
        client.settimeout(None)
        dt = time.time() - t0
        log("#%d %s -> %s (%.2fs)" % (cid, sni, ip, dt),
            key="ok:" + sni, every=1 if dt > 2 else 50)
        t1 = threading.Thread(target=pump, args=(client, upstream), daemon=True)
        t2 = threading.Thread(target=pump, args=(upstream, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except Exception as e:
        log("#%d FAILED: %s: %s" % (cid, type(e).__name__, e), key="fail", every=20)
    finally:
        for s in (client, upstream):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((LISTEN_HOST, LISTEN_PORT))
    except OSError as e:
        log("bind %s:%d failed: %s" % (LISTEN_HOST, LISTEN_PORT, e))
        sys.exit(1)
    srv.listen(128)
    log("relay on %s:%d | 运行时 DNS + 赢家缓存 + %ss 探测超时"
        % (LISTEN_HOST, LISTEN_PORT, PROBE_TIMEOUT))
    while True:
        try:
            client, _ = srv.accept()
            threading.Thread(target=handle, args=(client,), daemon=True).start()
        except KeyboardInterrupt:
            break
        except Exception as e:
            log("accept error: %s" % e, key="accept", every=20)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
