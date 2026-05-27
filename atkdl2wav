#!/usr/bin/env python3
"""
atkdl2wav — 把正点原子逻辑分析仪录的 .atkdl 文件转成能听的 WAV 音频。

用法:
  python3 pwm2wav.py -i 录音.atkdl
  python3 pwm2wav.py -i 录音.atkdl -o 输出.wav -t 8 -l 28 -r 29
"""

import argparse
import os
import struct
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

PWM_CARRIER = 500_000


def extract_duty(data: bytes) -> list:
    """从 bit-packed 字节流中检测 PWM 边沿，返回占空比列表。"""
    transitions = []
    cur = 0
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0xFF:
            if cur == 0:
                transitions.append(i * 8)
                cur = 1
            i += 1
        elif b == 0x00:
            if cur == 1:
                transitions.append(i * 8)
                cur = 0
            i += 1
        else:
            for bit in range(8):
                val = (b >> bit) & 1
                if val != cur:
                    transitions.append(i * 8 + bit)
                    cur = val
            i += 1

    result = []
    for j in range(1, len(transitions) - 1, 2):
        hw = transitions[j] - transitions[j - 1]
        lw = transitions[j + 1] - transitions[j]
        period = hw + lw
        if 1000 < period < 3000:
            result.append(hw / period)
    return result


def list_bins(zf: zipfile.ZipFile, ch: int) -> list:
    """在 zip 里找指定通道的 .bin 文件，按 (段号, 块号) 排序返回。"""
    prefix = f"{ch}/"
    files = []
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith(".bin"):
            continue
        base = name[len(prefix):].replace(".bin", "")
        parts = base.split("-")
        if len(parts) == 2:
            try:
                files.append((int(parts[0]), int(parts[1]), name))
            except ValueError:
                pass
    files.sort()
    return files


def channel_has_bins(zf: zipfile.ZipFile, ch: int) -> bool:
    """快速检查通道是否存在，不排序。"""
    prefix = f"{ch}/"
    for name in zf.namelist():
        if name.startswith(prefix) and name.endswith(".bin"):
            return True
    return False


def read_channel(zf: zipfile.ZipFile, ch: int, workers: int) -> list:
    files = list_bins(zf, ch)
    total = len(files)
    if total == 0:
        return []

    duties = [None] * total
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_duty, zf.read(name)): i
                   for i, (_, _, name) in enumerate(files)}
        for f in as_completed(futures):
            idx = futures[f]
            duties[idx] = f.result()
            done += 1
            if done % 50 == 0 or done == total:
                pct = done * 100 // total
                bar = "#" * (pct * 40 // 100) + "-" * (40 - pct * 40 // 100)
                sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({done}/{total})")
                sys.stdout.flush()
    print()

    flat = []
    for d in duties:
        if d:
            flat.extend(d)
    return flat


def write_wav32(path: str, left: list, right: list, rate: int):
    n = min(len(left), len(right))
    with open(path, "wb") as f:
        sz = n * 8
        f.write(b"RIFF" + struct.pack("<I", 36 + sz) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 3, 2, rate, rate * 8, 8, 32))
        f.write(b"data" + struct.pack("<I", sz))
        # 批量打包：先构造 bytes 再一次性写入
        samples = bytearray(n * 8)
        for i in range(n):
            struct.pack_into("<ff", samples, i * 8,
                             (left[i] - 0.5) * 5.0,
                             (right[i] - 0.5) * 5.0)
        f.write(samples)
    print(f"  {n} samples @ {rate}Hz = {n / rate:.2f}s, 32-bit float stereo")
    print(f"  -> {path}")


def main():
    p = argparse.ArgumentParser(
        description="把正点原子逻辑分析仪录的 .atkdl 文件转成 WAV 音频。",
        epilog="默认左声道=通道28(pin10)，右声道=通道29(pin9)。\n"
               "输出 32-bit float WAV，采样率 ~500kHz，不会产生临时文件。\n\n"
               "示例:\n"
               "  python3 pwm2wav.py -i 录音.atkdl\n"
               "  python3 pwm2wav.py -i 录音.atkdl -o 输出.wav -t 8\n"
               "  python3 pwm2wav.py -i 录音.atkdl -l 0 -r 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", help="输入的 .atkdl 文件")
    p.add_argument("-o", "--output", help="输出的 .wav 文件（默认与输入同名）")
    p.add_argument("-t", "--threads", type=int, default=2,
                   help="并行处理的进程数（默认 2）")
    p.add_argument("-l", "--left", type=int, default=28,
                   help="左声道对应的逻辑分析仪通道号（默认 28，即 pin10）")
    p.add_argument("-r", "--right", type=int, default=29,
                   help="右声道对应的逻辑分析仪通道号（默认 29，即 pin9）")
    args = p.parse_args()

    if not args.input:
        p.print_help()
        print(f"\n示例: python3 {sys.argv[0]} -i 你的文件.atkdl")
        sys.exit(0)

    if not os.path.isfile(args.input):
        print(f"错误: 找不到文件 {args.input}", file=sys.stderr)
        sys.exit(1)

    output = args.output or (os.path.splitext(args.input)[0] + ".wav")
    workers = max(1, args.threads)

    print(f"输入:   {args.input}")
    print(f"输出:   {output}")
    print(f"进程数: {workers}")
    print(f"左声道: 通道{args.left}    右声道: 通道{args.right}")

    try:
        zf = zipfile.ZipFile(args.input, "r")
    except zipfile.BadZipFile:
        print(f"错误: {args.input} 不是合法的 zip/atkdl 文件", file=sys.stderr)
        sys.exit(1)

    with zf:
        for ch in (args.left, args.right):
            if not channel_has_bins(zf, ch):
                print(f"错误: 文件里没有通道 {ch} 的数据", file=sys.stderr)
                sys.exit(1)

        print(f"\n通道{args.left}（左声道）:")
        left = read_channel(zf, args.left, workers)
        print(f"  {len(left)} 个 PWM 周期")

        print(f"\n通道{args.right}（右声道）:")
        right = read_channel(zf, args.right, workers)
        print(f"  {len(right)} 个 PWM 周期")

    print("\n写入 WAV...")
    write_wav32(output, left, right, PWM_CARRIER)
    print("完成！")


if __name__ == "__main__":
    main()
