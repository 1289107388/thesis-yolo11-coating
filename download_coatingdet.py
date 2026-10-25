# -*- coding: utf-8 -*-
"""CoatingDet_v2.zip 多连接分段下载器（纯标准库，支持断点续传与完成后 MD5 校验）

用法: python -u download_coatingdet.py
进度: 直接看控制台输出（后台运行时看任务输出文件）
"""
import hashlib
import json
import os
import threading
import time
import urllib.request

URL = "https://zenodo.org/api/records/21600391/files/CoatingDet_v2.zip/content"
RECORD_API = "https://zenodo.org/api/records/21600391"
TOTAL = 15_203_708_499          # 与 Zenodo 记录一致
CHUNK = 8 * 1024 * 1024         # 8 MB / 块
WORKERS = int(os.environ.get("COATINGDET_WORKERS", "4"))  # 并发过高会被 Zenodo 403
BASE = r"D:\毕业论文"
FINAL = os.path.join(BASE, "CoatingDet_v2.zip")
DONE_DIR = os.path.join(BASE, "__chunks_done")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*",
}

N_CHUNKS = (TOTAL + CHUNK - 1) // CHUNK
lock = threading.Lock()
done_set = set()                # 已完成块号（含历史续传）
counter = [0]                   # 本次会话累计下载字节


def marker_path(i):
    return os.path.join(DONE_DIR, f"{i:05d}.done")


def chunk_size(i):
    return min(CHUNK, TOTAL - i * CHUNK)


def fetch_official_md5():
    try:
        req = urllib.request.Request(RECORD_API, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as r:
            for f in json.loads(r.read().decode("utf-8"))["files"]:
                if f.get("key") == "CoatingDet_v2.zip":
                    c = f.get("checksum", "")
                    return c.split(":")[-1].lower() if ":" in c else c.lower()
    except Exception as e:
        print(f"[warn] 获取官方 MD5 失败（不影响下载，仅跳过校验）: {e}", flush=True)
    return ""


def worker(wid, queue):
    fd = open(FINAL, "r+b")
    try:
        while True:
            with lock:
                if not queue:
                    return
                i = queue.pop(0)
            cs = i * CHUNK
            size = chunk_size(i)
            pos, attempt = 0, 0
            while pos < size:
                try:
                    req = urllib.request.Request(
                        URL, headers={**HEADERS, "Range": f"bytes={cs + pos}-{cs + size - 1}"})
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        if resp.status != 206:
                            raise IOError(f"非 206 响应: {resp.status}")
                        while pos < size:
                            data = resp.read(65536)
                            if not data:
                                raise IOError("连接中断")
                            fd.seek(cs + pos)
                            fd.write(data)
                            pos += len(data)
                            with lock:
                                counter[0] += len(data)
                    attempt = 0
                except Exception as e:
                    attempt += 1
                    wait = min(60, 2 ** min(attempt, 6))
                    if "403" in str(e):          # IP 被限流，冷却后再试
                        wait = max(wait, 45)
                    print(f"[w{wid}] 块{i} @{pos}/{size}MB 第{attempt}次重试 "
                          f"{type(e).__name__}: {e}，{wait}s后续传", flush=True)
                    time.sleep(wait)
            open(marker_path(i), "wb").close()
            with lock:
                done_set.add(i)
    finally:
        fd.close()


def reporter(stop):
    last, last_t = 0, time.time()
    while not stop.is_set():
        stop.wait(60)
        with lock:
            cur, done_n = counter[0], len(done_set)
        now = time.time()
        spd = (cur - last) / max(now - last_t, 1e-3)
        last, last_t = cur, now
        done_bytes = sum(chunk_size(i) for i in done_set)
        pct = 100.0 * done_bytes / TOTAL
        eta_h = (TOTAL - done_bytes) / spd / 3600 if spd > 1 else float("inf")
        print(f"[{time.strftime('%H:%M:%S')}] {done_n}/{N_CHUNKS} 块 | "
              f"{done_bytes / 1e9:.2f}/{TOTAL / 1e9:.2f} GB ({pct:.1f}%) | "
              f"{spd / 1024:.0f} KB/s | 预计剩余 {eta_h:.1f} h", flush=True)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    os.makedirs(DONE_DIR, exist_ok=True)
    if not os.path.exists(FINAL) or os.path.getsize(FINAL) != TOTAL:
        with open(FINAL, "wb") as f:
            f.truncate(TOTAL)
        print(f"已预分配文件 {TOTAL} 字节", flush=True)

    historic = set()
    for name in os.listdir(DONE_DIR):
        if name.endswith(".done"):
            historic.add(int(name.split(".")[0]))
    with lock:
        done_set.update(historic)
    queue = [i for i in range(N_CHUNKS) if i not in done_set]
    print(f"总块数 {N_CHUNKS}，已完成 {len(historic)}，待下载 {len(queue)}，"
          f"并发 {WORKERS}", flush=True)

    stop = threading.Event()
    threading.Thread(target=reporter, args=(stop,), daemon=True).start()
    threads = [threading.Thread(target=worker, args=(w, queue)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()

    missing = [i for i in range(N_CHUNKS) if i not in done_set]
    if missing:
        print(f"[错误] 仍有 {len(missing)} 块未完成: {missing[:10]}...，重新运行脚本续传", flush=True)
        raise SystemExit(1)

    size = os.path.getsize(FINAL)
    print(f"全部块完成，文件大小 {size}（应为 {TOTAL}）", flush=True)
    if size != TOTAL:
        print("[错误] 大小不符", flush=True)
        raise SystemExit(1)

    official = fetch_official_md5()
    if not official:
        print("[跳过] 未取得官方 MD5，请到 Zenodo 页面手动核对", flush=True)
        return
    print("开始 MD5 校验（15 GB 全量读取，约几分钟）...", flush=True)
    local = md5_of(FINAL)
    print(f"本地  MD5: {local}", flush=True)
    print(f"官方  MD5: {official}", flush=True)
    print("VERIFY_OK" if local == official else "VERIFY_FAIL 请重新运行脚本（会只重下损坏块需先清空 __chunks_done）",
          flush=True)


if __name__ == "__main__":
    main()
