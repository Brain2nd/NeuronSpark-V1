"""
磁盘空间监控：当 /workspace 剩余空间 < 100GB 时自动删除最早的 checkpoint。

按文件系统时间戳（mtime）排序，每次删除最早的一个 checkpoint 目录，
直到剩余空间 >= 100GB。

用法：
    # 后台常驻
    nohup python3 scripts/disk_monitor.py > /dev/null 2>&1 &

    # 自定义参数
    python3 scripts/disk_monitor.py --checkpoint_dir checkpoints --threshold_gb 100 --interval 60
"""

import os
import sys
import time
import shutil
import argparse


def get_free_gb(path='/workspace'):
    """获取指定路径所在磁盘的剩余空间（GB）。"""
    st = os.statvfs(path)
    return (st.f_bavail * st.f_frsize) / (1024 ** 3)


def get_checkpoint_dirs(checkpoint_dir):
    """获取所有 checkpoint 目录，按 mtime 升序排列（最早的在前）。"""
    if not os.path.exists(checkpoint_dir):
        return []
    entries = []
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and name.startswith('ckpt_step'):
            mtime = os.path.getmtime(path)
            entries.append((mtime, path))
    entries.sort()  # 按 mtime 升序
    return entries


def main():
    parser = argparse.ArgumentParser(description="磁盘空间监控，自动清理早期 checkpoint")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='checkpoint 目录路径')
    parser.add_argument('--threshold_gb', type=float, default=100.0,
                        help='剩余空间阈值（GB），低于此值触发清理')
    parser.add_argument('--min_keep', type=int, default=2,
                        help='至少保留的 checkpoint 数量')
    parser.add_argument('--interval', type=int, default=60,
                        help='检查间隔（秒）')
    parser.add_argument('--disk_path', type=str, default='/workspace',
                        help='监控的磁盘路径')
    args = parser.parse_args()

    print(f"磁盘监控启动: 监控 {args.disk_path}, 阈值 {args.threshold_gb}GB, "
          f"checkpoint 目录 {args.checkpoint_dir}, 间隔 {args.interval}s")
    sys.stdout.flush()

    while True:
        free_gb = get_free_gb(args.disk_path)

        if free_gb < args.threshold_gb:
            entries = get_checkpoint_dirs(args.checkpoint_dir)
            if len(entries) <= args.min_keep:
                print(f"[WARN] 剩余 {free_gb:.1f}GB < {args.threshold_gb}GB，"
                      f"但仅剩 {len(entries)} 个 checkpoint，不再删除")
                sys.stdout.flush()
            else:
                while free_gb < args.threshold_gb and len(entries) > args.min_keep:
                    mtime, oldest = entries.pop(0)
                    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
                    dir_size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fns in os.walk(oldest) for f in fns
                    ) / (1024 ** 3)
                    shutil.rmtree(oldest)
                    free_gb = get_free_gb(args.disk_path)
                    print(f"[CLEAN] 删除 {oldest} (创建于 {mtime_str}, {dir_size:.1f}GB), "
                          f"剩余空间 {free_gb:.1f}GB")
                    sys.stdout.flush()

        time.sleep(args.interval)


if __name__ == '__main__':
    main()
