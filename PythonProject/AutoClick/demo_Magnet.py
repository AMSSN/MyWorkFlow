"""
磁力链接解析器

依赖:  pip install libtorrent

四个步骤对应:
  1. MagnetValidator / MagnetParser.prepare()  -> 校验合法性 + 判断网络环境 (后续步骤的基础)
  2. MagnetParser.fetch_files()               -> 联网拉取元数据, 获取文件列表
  3. MagnetParser.filter_files()              -> 依据策略对文件列表做筛选
  4. SizeFilter / TypeFilter / NameRegexFilter / CompositeFilter -> 不同筛选策略, 可自由组合
"""

from __future__ import annotations

import re
import time
import socket
import tempfile
import warnings
from dataclasses import dataclass
from typing import List, Optional

import libtorrent as lt


@dataclass
class TorrentFile:
    """表示一个种子内的文件及其元数据"""
    index: int       # 在种子中的序号
    path: str        # 完整路径
    name: str        # 文件名
    size: int        # 字节

    @property
    def ext(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""

    @staticmethod
    def human(size: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        f = float(size)
        for u in units:
            if f < 1024 or u == "TB":
                return f"{f:.2f} {u}"
            f /= 1024
        return f"{size} B"


# ---------------------------------------------------------------------------
# 步骤 1: 合法性校验 + 网络环境判断
# ---------------------------------------------------------------------------
class MagnetValidator:
    # xt=urn:btih:<hash>, hash 为 40 位 hex(SHA-1) 或 32 位 base32
    _RE = re.compile(
        r"xt=urn:btih:([A-Za-z0-9]{40}|[A-Za-z2-7]{32})(?:&|$)",
        re.IGNORECASE,
    )

    @classmethod
    def is_valid(cls, magnet: str) -> bool:
        if not magnet or not magnet.strip().startswith("magnet:?"):
            return False
        return bool(cls._RE.search(magnet))

    @classmethod
    def info_hash(cls, magnet: str) -> Optional[str]:
        m = cls._RE.search(magnet or "")
        return m.group(1).upper() if m else None

    @staticmethod
    def check_network(timeout: float = 5.0) -> bool:
        """探测外网基础连通性 (磁力主要靠 UDP/DHT, 此处用 TCP 节点做前置判断)"""
        probes = [
            ("tracker.opentrackr.org", 1337),
            ("1.1.1.1", 53),
            ("8.8.8.8", 53),
        ]
        for host, port in probes:
            try:
                with socket.create_connection((host, port), timeout=timeout):
                    return True
            except OSError:
                continue
        return False


# ---------------------------------------------------------------------------
# 筛选策略 (步骤 4)  —— 可单独使用, 也可用 CompositeFilter 组合
# ---------------------------------------------------------------------------
class FilterStrategy:
    def apply(self, files: List[TorrentFile]) -> List[TorrentFile]:
        raise NotImplementedError


class SizeFilter(FilterStrategy):
    """按体积筛选 (字节). max_size=None 表示不限制上限"""
    def __init__(self, min_size: int = 0, max_size: Optional[int] = None):
        self.min_size = min_size
        self.max_size = max_size

    def apply(self, files: List[TorrentFile]) -> List[TorrentFile]:
        hi = self.max_size if self.max_size is not None else float("inf")
        return [f for f in files if self.min_size <= f.size <= hi]


class TypeFilter(FilterStrategy):
    """按扩展名筛选. include=True 保留匹配项, False 排除匹配项"""
    def __init__(self, extensions: List[str], include: bool = True):
        self.exts = {e.lower().lstrip(".") for e in extensions}
        self.include = include

    def apply(self, files: List[TorrentFile]) -> List[TorrentFile]:
        res = []
        for f in files:
            hit = f.ext in self.exts
            if hit == self.include:
                res.append(f)
        return res


class NameRegexFilter(FilterStrategy):
    """按文件名正则筛选 (匹配文件名, 非路径)"""
    def __init__(self, pattern: str, flags: int = 0):
        self.rx = re.compile(pattern, flags)

    def apply(self, files: List[TorrentFile]) -> List[TorrentFile]:
        return [f for f in files if self.rx.search(f.name)]


class CompositeFilter(FilterStrategy):
    """组合多个策略, 取交集 (AND)"""
    def __init__(self, *strategies: FilterStrategy):
        self.strategies = list(strategies)

    def apply(self, files: List[TorrentFile]) -> List[TorrentFile]:
        result = files
        for s in self.strategies:
            result = s.apply(result)
        return result


# ---------------------------------------------------------------------------
# 主解析器
# ---------------------------------------------------------------------------
class MagnetParser:
    def __init__(
        self,
        magnet: str,
        timeout: int = 60,
        save_path: Optional[str] = None,
    ):
        if not MagnetValidator.is_valid(magnet):
            raise ValueError(f"非法磁力链接: {magnet}")
        self.magnet = magnet.strip()
        self.timeout = timeout
        self.save_path = save_path or tempfile.gettempdir()
        self._torrent_info = None
        self.files: List[TorrentFile] = []

    # 步骤 1: 校验合法性 + 判断网络环境 (后续步骤的基础)
    def prepare(self) -> dict:
        return {
            "valid": True,
            "info_hash": MagnetValidator.info_hash(self.magnet),
            "network_available": MagnetValidator.check_network(),
        }

    # 步骤 2: 联网拉取元数据 -> 文件列表
    def fetch_files(self, require_network: bool = True) -> List[TorrentFile]:
        if require_network and not MagnetValidator.check_network():
            raise RuntimeError("网络不可用, 无法获取元数据")

        ses = lt.session()
        try:
            # 2.x 起 DHT 由 session 自动启动, start_dht/add_dht_router 已弃用;
            # 用 warnings 抑制可能的 DeprecationWarning, 旧版本仍可照常调用。
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    ses.start_dht()
                except Exception:
                    pass
                try:
                    ses.add_dht_router("router.bittorrent.com", 6881)
                    ses.add_dht_router("dht.transmissionbt.com", 6881)
                except Exception:
                    pass

            handle = self._add_magnet(ses)
            deadline = time.time() + self.timeout
            while not self._has_metadata(handle):
                if time.time() > deadline:
                    ses.remove_torrent(handle)
                    raise TimeoutError(
                        "获取元数据超时, 请检查 DHT 网络或换用带 tracker 的磁力链接"
                    )
                time.sleep(0.3)

            self._torrent_info = handle.get_torrent_info()
            self.files = self._extract_files(self._torrent_info)
            ses.remove_torrent(handle)
        finally:
            del ses
        return self.files

    @staticmethod
    def libtorrent_version() -> str:
        """返回已安装的 libtorrent 版本字符串, 用于排查兼容性"""
        return str(getattr(lt, "__version__", "unknown"))

    def _add_magnet(self, ses):
        """兼容新旧 libtorrent 添加磁力的方式。

        新版(2.0+): session.add_torrent(parse_magnet_uri(...))
        旧版(1.x): lt.add_magnet_uri(session, magnet_uri_string, dict)
        """
        # 构造 params (新 API 优先)
        params = lt.parse_magnet_uri(self.magnet)
        params.save_path = self.save_path
        try:  # 只拉元数据, 不下载文件实体
            params.flags |= lt.torrent_flags.only_metadata
        except Exception:
            pass

        # 优先使用 session.add_torrent (替代已弃用的 add_magnet_uri)
        try:
            return ses.add_torrent(params)
        except Exception as e:
            # 旧版 libtorrent 不支持 session.add_torrent(params) 时,
            # 回退到弃用的 add_magnet_uri(session, uri, dict)
            if "did not match C++ signature" not in str(e) and "ArgumentError" not in str(e):
                raise
            add_dict = {"save_path": self.save_path}
            try:
                add_dict["flags"] = int(getattr(lt.torrent_flags, "only_metadata", 0))
            except Exception:
                pass
            return lt.add_magnet_uri(ses, self.magnet, add_dict)

    @staticmethod
    def _has_metadata(handle) -> bool:
        """判断是否已拿到元数据, 兼容新旧 libtorrent API。

        新版(2.1+): has_metadata() 已弃用, 改用 status().has_metadata
        旧版(1.x): handle.has_metadata()
        """
        try:
            return bool(handle.status().has_metadata)
        except Exception:
            try:
                return handle.has_metadata()
            except Exception:
                # 最兜底: 直接判断 torrent_file 是否存在
                try:
                    return handle.torrent_file() is not None
                except Exception:
                    return False

    @staticmethod
    def _extract_files(ti) -> List[TorrentFile]:
        fs = ti.files()
        if hasattr(fs, "num_files"):           # libtorrent 2.x
            entries = [fs.at(i) for i in range(fs.num_files())]
        else:                                  # libtorrent 1.x
            entries = list(fs)
        out = []
        for i, f in enumerate(entries):
            path = f.path
            name = path.replace("\\", "/").split("/")[-1]
            out.append(TorrentFile(index=i, path=path, name=name, size=int(f.size)))
        return out

    # 步骤 3 + 4: 依据筛选策略过滤已获取的文件列表
    def filter_files(self, strategy: FilterStrategy) -> List[TorrentFile]:
        if not self.files:
            self.fetch_files()
        return strategy.apply(self.files)


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 换成真实磁力链接
    magnet = "magnet:?xt=urn:btih:08A45C8E55A2E25605ABE0EE54A684208A2D4CF6"

    print("libtorrent 版本:", MagnetParser.libtorrent_version())
    parser = MagnetParser(magnet)

    # 步骤 1: 合法性 + 网络环境
    env = parser.prepare()
    print("环境检查:", env)
    if not env["network_available"]:
        print("网络不可用, 终止。")
    else:
        # 步骤 2: 获取文件列表
        files = parser.fetch_files()
        print(f"\n共 {len(files)} 个文件:")
        for f in files:
            print(f"  [{f.index}] {f.name}  {TorrentFile.human(f.size)}")

        # 步骤 3 + 4: 组合筛选策略
        # 例: 体积 >= 100MB 且 类型为 mp4 / mkv
        strategy = CompositeFilter(
            SizeFilter(min_size=100 * 1024 * 1024)
        )
        selected = parser.filter_files(strategy)
        print("\n筛选结果 (>=100MB:")
        for f in selected:
            print(f"  {f.name}  {TorrentFile.human(f.size)}")

        # 其它策略示例:
        # TypeFilter(["txt", "nfo"], include=False)        # 排除文本/说明文件
        # NameRegexFilter(r"(?i)sample|trailer")            # 文件名含 sample/trailer
        # CompositeFilter(SizeFilter(max_size=10*1024*1024), TypeFilter(["mkv"]))
