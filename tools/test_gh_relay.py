#!/usr/bin/env python3
"""gh_relay.py 的离线单测（不联网）。

运行：
    python tools/test_gh_relay.py
"""
from __future__ import annotations

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gh_relay  # noqa: E402


def make_client_hello(host: str) -> bytes:
    """手工拼一个带 SNI 的 TLS ClientHello（结构与 Chrome 一致，够 parse_sni 解析）。"""
    name = host.encode("ascii")
    sni_ext = (
        b"\x00\x00"                                  # extension type: server_name
        + (len(name) + 5).to_bytes(2, "big")         # extension length
        + (len(name) + 3).to_bytes(2, "big")         # server_name_list length
        + b"\x00"                                    # name_type: host_name
        + len(name).to_bytes(2, "big")
        + name
    )
    body = (
        b"\x03\x03"                                  # legacy_version TLS 1.2
        + b"\x00" * 32                               # random
        + b"\x00"                                    # session id length
        + b"\x00\x02\x13\x01"                        # cipher suites
        + b"\x01\x00"                                # compression methods
        + len(sni_ext).to_bytes(2, "big") + sni_ext  # extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + len(handshake).to_bytes(2, "big") + handshake


class TestSniParsing(unittest.TestCase):
    def test_parses_sni_from_client_hello(self):
        hello = make_client_hello("github.com")
        self.assertEqual(gh_relay.parse_sni(hello), "github.com")

    def test_rejects_non_handshake(self):
        self.assertIsNone(gh_relay.parse_sni(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"))
        self.assertIsNone(gh_relay.parse_sni(b""))

    def test_name_encoding_roundtrip(self):
        encoded = gh_relay._encode_name("github.com")
        self.assertEqual(encoded, b"\x06github\x03com\x00")
        self.assertEqual(gh_relay._skip_name(encoded + b"tail", 0), len(encoded))


class TestServerHelloCheck(unittest.TestCase):
    def test_accepts_server_hello_header(self):
        # TLS record type 0x16 (handshake), handshake type 0x02 (ServerHello)
        self.assertTrue(gh_relay.is_server_hello(b"\x16\x03\x03\x00\x30\x02" + b"\x00" * 16))

    def test_rejects_alert_and_short_prefix(self):
        self.assertFalse(gh_relay.is_server_hello(b"\x15\x03\x03\x00\x02\x02\x28"))
        self.assertFalse(gh_relay.is_server_hello(b"\x16\x03"))
        self.assertFalse(gh_relay.is_server_hello(b""))


class TestCandidates(unittest.TestCase):
    def setUp(self):
        self._saved_dns = gh_relay.dns_a
        gh_relay.dns_a = lambda host: ["9.9.9.9"]      # 离线：不真的查 DNS
        gh_relay._winner.clear()
        gh_relay._dns_cache.clear()

    def tearDown(self):
        gh_relay.dns_a = self._saved_dns
        gh_relay._winner.clear()
        gh_relay._dns_cache.clear()

    def test_order_is_winner_preferred_dns(self):
        gh_relay._winner["github.com"] = ("1.2.3.4", time.time())
        ips = gh_relay.candidates("github.com")
        self.assertEqual(ips[0], "1.2.3.4")
        self.assertEqual(ips[1], gh_relay.PREFERRED["github.com"][0])
        self.assertEqual(ips[-1], "9.9.9.9")
        self.assertEqual(len(ips), len(set(ips)))

    def test_stale_winner_expires(self):
        gh_relay._winner["github.com"] = ("1.2.3.4", time.time() - gh_relay.WINNER_TTL - 1)
        ips = gh_relay.candidates("github.com")
        self.assertNotIn("1.2.3.4", ips)
        self.assertNotIn("github.com", gh_relay._winner)


if __name__ == "__main__":
    unittest.main(verbosity=2)
