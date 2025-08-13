from __future__ import annotations

import os
import struct
import sys
from typing import Optional, List


def tokenize(s: str, delim: str) -> List[str]:
    # Keep for compatibility; used sparingly after optimized parser
    return s.split(delim)


class TimeNode:
    __slots__ = ("timestamp", "arbitrary_node_info", "left", "right", "height")

    def __init__(self, t: int, info: str) -> None:
        self.timestamp: int = int(t)
        self.arbitrary_node_info: str = info
        self.left: Optional[TimeNode] = None
        self.right: Optional[TimeNode] = None
        self.height: int = 1

    # Convenience for printing/debugging (not used by algorithms)
    def __repr__(self) -> str:
        return f"TimeNode(ts={self.timestamp}, info={self.arbitrary_node_info!r}, h={self.height})"


class TimeTree:
    def __init__(self, filename: Optional[str] = None) -> None:
        # Public root like the C++ header
        self.m_root: Optional[TimeNode] = None

        if filename is None:
            return

        # Mirror C++: if file is openable in binary, call load() and return (do not assign), else build from text
        try:
            with open(filename, "rb"):
                # Call load but ignore the returned tree (C++ ignores it and returns)
                TimeTree.load(filename)
                return
        except OSError:
            # Not openable as binary; build from text.
            pass
        self.m_root = self.buildAVLTree(filename)

    # ----- Public API matching Python usage wrappers -----
    def get(self, timestamp: int, threshold: int) -> Optional[TimeNode]:
        # threshold must be in same units as timestamp
        return self._findClosest(self.m_root, int(timestamp), int(threshold))

    def save(self, binfile: str) -> bool:
        try:
            with open(binfile, "wb") as f:
                self._save_node(f, self.m_root)
            return True
        except OSError:
            return False

    @staticmethod
    def load(binfile: str) -> Optional["TimeTree"]:
        try:
            with open(binfile, "rb") as f:
                tree = TimeTree()
                tree.m_root = TimeTree._load_node(f)
                return tree
        except OSError:
            return None

    # Expose helpers; default to root only on initial call. Recursion uses private helpers.
    def getHeight(self, node: Optional[TimeNode] = None) -> int:
        return node.height if node is not None else 0

    def getBalanceFactor(self, node: Optional[TimeNode] = None) -> int:
        if node is None:
            return 0
        return self.getHeight(node.left) - self.getHeight(node.right)

    def countLeafNodes(self, node: Optional[TimeNode] = None) -> int:
        n = self.m_root if node is None else node
        return self._countLeafNodes(n)

    def _countLeafNodes(self, n: Optional[TimeNode]) -> int:
        if n is None:
            return 0
        if n.left is None and n.right is None:
            return 1
        return self._countLeafNodes(n.left) + self._countLeafNodes(n.right)

    def getTreeDepth(self, node: Optional[TimeNode] = None) -> int:
        n = self.m_root if node is None else node
        return self._getTreeDepth(n)

    def _getTreeDepth(self, n: Optional[TimeNode]) -> int:
        if n is None:
            return 0
        return 1 + max(self._getTreeDepth(n.left), self._getTreeDepth(n.right))

    def getTotalNodes(self, node: Optional[TimeNode] = None) -> int:
        n = self.m_root if node is None else node
        return self._getTotalNodes(n)

    def _getTotalNodes(self, n: Optional[TimeNode]) -> int:
        if n is None:
            return 0
        return 1 + self._getTotalNodes(n.left) + self._getTotalNodes(n.right)

    def appendAVLTree(self, timestamp_filepath: str) -> None:
        self.m_root = self.buildAVLTree(timestamp_filepath, self.m_root)

    def buildAVLTree(self, timestamp_filepath: str, root: Optional[TimeNode] = None) -> Optional[TimeNode]:
        if not os.path.exists(timestamp_filepath):
            print(f"ERROR: Cannot open file: {timestamp_filepath}")
            return None

        r = root
        with open(timestamp_filepath, "r", encoding="utf-8", errors="ignore") as file:
            for raw in file:
                # C++ getline removes only the trailing newline; preserve other spaces
                line = raw.rstrip("\r\n")
                # Fast parse: take first token before first space
                sp = line.find(' ')
                first = line if sp == -1 else line[:sp]
                # Expect pattern: PREFIX_<TIMESTAMP>[_FRAME]
                # Find first '_' and second '_'
                u1 = first.find('_')
                if u1 == -1:
                    print(f"Skipping malformed line: {line}")
                    continue
                u2 = first.find('_', u1 + 1)
                if u2 == -1:
                    ts_str = first[u1 + 1:]
                    frameidx = ""
                else:
                    ts_str = first[u1 + 1:u2]
                    frameidx = first[u2 + 1:]
                try:
                    ts = int(ts_str)
                except ValueError:
                    ts = 0
                if ts <= 0:
                    print(f"Invalid timestamp found: {ts}")
                    print(f"Skipping malformed line: {line}")
                    continue
                # Insert into AVL
                r = self._insert(r, ts, frameidx)
        return r

    # ----- Internal AVL helpers -----
    def _rotateRight(self, y: TimeNode) -> TimeNode:
        x = y.left
        assert x is not None  # rotateRight only called when left exists
        T2 = x.right

        x.right = y
        y.left = T2

        yl = y.left.height if y.left else 0
        yr = y.right.height if y.right else 0
        y.height = (yl if yl > yr else yr) + 1
        xl = x.left.height if x.left else 0
        xr = x.right.height if x.right else 0
        x.height = (xl if xl > xr else xr) + 1
        return x

    def _rotateLeft(self, x: TimeNode) -> TimeNode:
        y = x.right
        assert y is not None  # rotateLeft only called when right exists
        T2 = y.left

        y.left = x
        x.right = T2

        xl = x.left.height if x.left else 0
        xr = x.right.height if x.right else 0
        x.height = (xl if xl > xr else xr) + 1
        yl = y.left.height if y.left else 0
        yr = y.right.height if y.right else 0
        y.height = (yl if yl > yr else yr) + 1
        return y

    def _insert(self, root: Optional[TimeNode], timestamp: int, frameidx: str) -> TimeNode:
        if root is None:
            return TimeNode(timestamp, frameidx)

        if timestamp < root.timestamp:
            root.left = self._insert(root.left, timestamp, frameidx)
        elif timestamp > root.timestamp:
            root.right = self._insert(root.right, timestamp, frameidx)
        else:
            return root

        lh = root.left.height if root.left else 0
        rh = root.right.height if root.right else 0
        root.height = (lh if lh > rh else rh) + 1
        balance = lh - rh

        if balance > 1 and root.left and timestamp < root.left.timestamp:
            return self._rotateRight(root)
        if balance < -1 and root.right and timestamp > root.right.timestamp:
            return self._rotateLeft(root)
        if balance > 1 and root.left and timestamp > root.left.timestamp:
            root.left = self._rotateLeft(root.left)
            return self._rotateRight(root)
        if balance < -1 and root.right and timestamp < root.right.timestamp:
            root.right = self._rotateRight(root.right)
            return self._rotateLeft(root)

        return root

    def _findClosest(self, root: Optional[TimeNode], target: int, threshold: int) -> Optional[TimeNode]:
        closest: Optional[TimeNode] = None
        minDiff = 2**63 - 1  # max int64

        curr = root
        while curr is not None:
            diff = abs(curr.timestamp - target)
            if diff < minDiff:
                minDiff = diff
                closest = curr
            curr = curr.left if target < curr.timestamp else curr.right

        if closest is not None and minDiff <= threshold:
            return closest
        return None

    # ----- Serialization helpers (pre-order), matching C++ binary layout -----
    # Precompiled struct packers for speed (little-endian)
    _S_Q = struct.Struct('<Q')
    _S_q = struct.Struct('<q')
    _S_bb = struct.Struct('<??')

    @staticmethod
    def _save_node(f, node: Optional[TimeNode]) -> None:
        if node is None:
            return
        # int64_t timestamp
        f.write(TimeTree._S_q.pack(int(node.timestamp)))
        # size_t len (assume 64-bit little-endian)
        data = node.arbitrary_node_info.encode("utf-8")
        f.write(TimeTree._S_Q.pack(len(data)))
        # bytes of string
        if data:
            f.write(data)
        # bool has_left, bool has_right (1 byte each)
        has_left = node.left is not None
        has_right = node.right is not None
        f.write(TimeTree._S_bb.pack(has_left, has_right))
        # recurse
        if has_left:
            TimeTree._save_node(f, node.left)
        if has_right:
            TimeTree._save_node(f, node.right)

    @staticmethod
    def _load_node(f) -> Optional[TimeNode]:
        # Check EOF by trying to read timestamp
        hdr = f.read(8)
        if len(hdr) < 8:
            return None
        (timestamp,) = TimeTree._S_q.unpack(hdr)
        len_bytes = f.read(8)
        if len(len_bytes) < 8:
            return None
        (length,) = TimeTree._S_Q.unpack(len_bytes)
        info_bytes = f.read(length) if length > 0 else b""
        if len(info_bytes) < length:
            return None
        has_flags = f.read(2)
        if len(has_flags) < 2:
            return None
        has_left, has_right = TimeTree._S_bb.unpack(has_flags)

        node = TimeNode(int(timestamp), info_bytes.decode("utf-8"))
        if has_left:
            node.left = TimeTree._load_node(f)
        if has_right:
            node.right = TimeTree._load_node(f)
        # height is not serialized; recompute lazily upwards if needed
        lh = node.left.height if node.left else 0
        rh = node.right.height if node.right else 0
        node.height = (lh if lh > rh else rh) + 1
        return node
