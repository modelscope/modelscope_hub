"""Pure Python SHA256 prefix export for multipart uploads."""

from __future__ import annotations

import base64
import json
import struct

from .._multipart_protocol import MultipartError

_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_K = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)
_MASK = 0xFFFFFFFF


def _rotate(value: int, bits: int) -> int:
    return ((value >> bits) | (value << (32 - bits))) & _MASK


class SHA256State:
    """Incremental SHA256 with aligned internal-state export."""

    def __init__(self) -> None:
        self._h = list(_IV)
        self._tail = b""
        self._count = 0

    def _compress(self, block: bytes) -> None:
        words = list(struct.unpack(">16I", block))
        for index in range(16, 64):
            x, y = words[index - 15], words[index - 2]
            words.append(
                (
                    words[index - 16]
                    + (_rotate(x, 7) ^ _rotate(x, 18) ^ (x >> 3))
                    + words[index - 7]
                    + (_rotate(y, 17) ^ _rotate(y, 19) ^ (y >> 10))
                )
                & _MASK
            )
        a, b, c, d, e, f, g, h = self._h
        for index, word in enumerate(words):
            t1 = (
                h + (_rotate(e, 6) ^ _rotate(e, 11) ^ _rotate(e, 25)) + ((e & f) ^ (~e & g)) + _K[index] + word
            ) & _MASK
            t2 = ((_rotate(a, 2) ^ _rotate(a, 13) ^ _rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & _MASK
            a, b, c, d, e, f, g, h = (t1 + t2) & _MASK, a, b, c, (d + t1) & _MASK, e, f, g
        self._h = [(x + y) & _MASK for x, y in zip(self._h, (a, b, c, d, e, f, g, h))]

    def update(self, data: bytes) -> None:
        self._count += len(data)
        data = self._tail + data
        end = len(data) // 64 * 64
        for offset in range(0, end, 64):
            self._compress(data[offset : offset + 64])
        self._tail = data[end:]

    def copy(self) -> SHA256State:
        result = SHA256State()
        result._h, result._tail, result._count = list(self._h), self._tail, self._count
        return result

    def digest(self) -> bytes:
        result = self.copy()
        count = self._count
        result.update(b"\x80" + b"\x00" * ((55 - count) % 64) + struct.pack(">Q", count * 8))
        return struct.pack(">8I", *result._h)

    def export_state(self) -> tuple[int, tuple[int, ...]]:
        if self._tail:
            raise ValueError("Prefix must be 64-byte aligned")
        return self._count, tuple(self._h)


def prefix_header(offset: int, words: tuple[int, ...]) -> str:
    if offset < 0 or offset % 64 or len(words) != 8 or offset * 8 >= 1 << 64:
        raise MultipartError("Invalid SHA256 prefix boundary.")
    if any(type(word) is not int or not 0 <= word <= _MASK for word in words):
        raise MultipartError("Invalid SHA256 prefix words.")
    bits = offset * 8
    value = {"hash_type": "sha256", "Nl": str(bits & _MASK), "Nh": str(bits >> 32), "num": "0", "md_len": "32"}
    value.update({f"h{i}": str(word) for i, word in enumerate(words)})
    return base64.b64encode(json.dumps(value, separators=(",", ":")).encode("utf-8")).decode("ascii")
