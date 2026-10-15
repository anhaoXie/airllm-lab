#!/usr/bin/env python3
"""
SNI 分流 TLS 中继 / SNI-routing TLS relay  —  127.0.0.1:443

背景:
  hosts 文件里有一段 Steam++ 遗留的屏蔽块, 把 github.com / api.github.com /
  raw.githubusercontent.com / huggingface.co 等域名指向 127.0.0.1, 而 Steam++ 的
  本地代理已不再运行 -> 这些域名全部变成死地址。

本脚本在 127.0.0.1:443 监听, 读出 TLS ClientHello 的 SNI, 连到该域名的真实 IP,
把字节原样双向转发。SNI 原样透传, 所以证书校验照常通过。

要点:
  - 不需要管理员权限, 不修改任何系统文件
  - ROUTES 里每个域名配一组候选 IP, 逐个尝试 (带 ServerHello 探测), 实现故障转移
    —— 因为 GitHub 的部分 IP 在本网络会被 SNI 阻断 (TCP 能连上, TLS 握手被丢弃)
  - 只在本进程存活期间生效, 关掉即恢复原状

启动: python gh_relay.py
停止: Ctrl+C 或结束进程
"""
import socket
import threading
import time
import os
import sys

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 443
UPSTREAM_PORT = 443
CONNECT_TIMEOUT = 6          # 连接 + 等 ServerHello 的超时
MAX_HELLO = 16384

# 实测本网络可用 (2026-10): 每个域名一组候选 IP, 按优先级排列
ROUTES = {
    "github.com": [
        "140.82.113.3", "140.82.114.3", "140.82.116.3", "140.82.121.3",
        "140.82.114.4", "140.82.116.4", "140.82.121.4",
        "20.27.177.113", "20.200.245.247", "4.208.26.197", "4.237.22.38",
    ],
    "www.github.com": ["140.82.113.3", "140.82.114.3", "140.82.116.3"],
    "gist.github.com": ["140.82.113.3", "140.82.114.3"],
    "api.github.com": ["20.205.243.168", "140.82.113.6", "140.82.114.6"],
    "codeload.github.com": ["20.205.243.165", "140.82.113.10"],
    "raw.githubusercontent.com": [
        "185.199.108.133", "185.199.109.133", "185.199.110.133", "185.199.111.133",
    ],
    "objects.githubusercontent.com": [
        "185.199.108.133", "185.199.109.133", "185.199.110.133", "185.199.111.133",
    ],
    "raw.github.com": ["185.199.108.133", "185.199.109.133"],
    "gist.githubusercontent.com": ["185.199.108.133", "185.199.109.133"],
    "github.io": ["185.199.108.153", "185.199.109.153"],
    "huggingface.co": ["3.161.55.71", "3.161.55.100"],
}

LOG_PATH = os.environ.get("RELAY_LOG", "")
_lock = threading.Lock()
_counter = [0]


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    with _lock:
        print(line, flush=True)
        if LOG_PATH:
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


def read_client_hello(sock):
    """读到完整的 TLS record (ClientHello) 为止。"""
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
    """从 TLS ClientHello 取出 SNI 主机名。"""
    try:
        if len(data) < 5 or data[0] != 0x16:
            return None
        p = 5
        if data[p] != 0x01:
            return None
        p += 4
        p += 2
        p += 32
        p += 1 + data[p]                                    # session_id
        cs_len = int.from_bytes(data[p:p + 2], "big")
        p += 2 + cs_len                                     # cipher_suites
        p += 1 + data[p]                                    # compression_methods
        ext_total = int.from_bytes(data[p:p + 2], "big")
        p += 2
        end = p + ext_total
        while p + 4 <= end:
            et = int.from_bytes(data[p:p + 2], "big")
            el = int.from_bytes(data[p + 2:p + 4], "big")
            p += 4
            if et == 0x0000:                                # server_name
                q = p + 2 + 1                               # list len + name_type
                nl = int.from_bytes(data[q:q + 2], "big")
                return data[q + 2:q + 2 + nl].decode("ascii", "ignore")
            p += el
    except Exception:
        pass
    return None


def connect_upstream(head, ips):
    """依次尝试候选 IP; 用'能否收到 ServerHello'判断该 IP 是否被阻断。"""
    last_err = "no candidate"
    for ip in ips:
        up = None
        try:
            up = socket.create_connection((ip, UPSTREAM_PORT), timeout=CONNECT_TIMEOUT)
            up.sendall(head)
            up.settimeout(CONNECT_TIMEOUT)
            first = up.recv(65536)          # 期待 TLS ServerHello
            if not first:
                raise IOError("upstream closed before ServerHello")
            up.settimeout(None)
            return up, ip, first
        except Exception as e:
            last_err = "%s: %s" % (type(e).__name__, e)
            if up is not None:
                try:
                    up.close()
                except Exception:
                    pass
    raise IOError(last_err)


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
    with _lock:
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
            log("#%d no SNI parsed, dropping %dB" % (cid, len(head)))
            return
        ips = ROUTES.get(sni)
        if not ips:
            try:
                ips = [socket.getaddrinfo(sni, UPSTREAM_PORT,
                                          proto=socket.IPPROTO_TCP)[0][4][0]]
            except Exception as e:
                log("#%d %s cannot resolve: %s" % (cid, sni, e))
                return
        t0 = time.time()
        upstream, ip, first = connect_upstream(head, ips)
        client.sendall(first)
        client.settimeout(None)
        log("#%d %s -> %s  (%.2fs)" % (cid, sni, ip, time.time() - t0))
        t1 = threading.Thread(target=pump, args=(client, upstream), daemon=True)
        t2 = threading.Thread(target=pump, args=(upstream, client), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except Exception as e:
        log("#%d FAILED: %s: %s" % (cid, type(e).__name__, e))
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
    srv.listen(64)
    total = sum(len(v) for v in ROUTES.values())
    log("relay listening on %s:%d  (%d hosts / %d candidate IPs)"
        % (LISTEN_HOST, LISTEN_PORT, len(ROUTES), total))
    while True:
        try:
            client, _peer = srv.accept()
            threading.Thread(target=handle, args=(client,), daemon=True).start()
        except KeyboardInterrupt:
            log("interrupted, exiting")
            break
        except Exception as e:
            log("accept error: %s" % e)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
