#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import struct
import csv
from bisect import bisect_left

import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2

# ----------- helpers -----------
PF_FMT = {1: 'b', 2: 'B', 3: 'h', 4: 'H', 5: 'i', 6: 'I', 7: 'f', 8: 'd'}


def nearest_ts(ts_list, t_query):
    if not ts_list:
        return None, float('inf')
    i = bisect_left(ts_list, t_query)
    cand = []
    if i < len(ts_list):
        cand.append(ts_list[i])
    if i > 0:
        cand.append(ts_list[i - 1])
    best_ts, best_dt = None, float('inf')
    for ts in cand:
        dt = abs(ts - t_query)
        if dt < best_dt:
            best_dt = dt
            best_ts = ts
    return best_ts, best_dt


def _pc2_to_numpy_xyz_i(msg):
    # Try fast path using read_points
    try:
        has_i = any(getattr(f, 'name', '').lower() in ('intensity', 'intensities') for f in msg.fields)
        fields = ['x', 'y', 'z'] + (['intensity'] if has_i else [])
        pts = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
        if len(pts) == 0:
            arr = np.zeros((0, 4), dtype=np.float32)
        else:
            arr = np.asarray(pts, dtype=np.float32).reshape(-1, len(fields))
            if arr.shape[1] == 3:
                inten = np.ones((arr.shape[0], 1), dtype=np.float32)
                arr = np.concatenate([arr, inten], axis=1)
        return arr
    except Exception as e:
        print(f"[LIDAR] read_points 失败，尝试fallback解析: {e}")

    # Fallback parse raw buffer
    try:
        fdict = {f.name: f for f in msg.fields}
        for k in ('x', 'y', 'z'):
            if k not in fdict:
                print(f"[LIDAR] 缺少字段 {k}, 返回空点云"); return np.zeros((0, 4), dtype=np.float32)
        offx, offy, offz = fdict['x'].offset, fdict['y'].offset, fdict['z'].offset
        dtx, dty, dtz = fdict['x'].datatype, fdict['y'].datatype, fdict['z'].datatype
        has_i = ('intensity' in fdict) or ('intensities' in fdict)
        if 'intensity' in fdict:
            offi, dti = fdict['intensity'].offset, fdict['intensity'].datatype
        elif 'intensities' in fdict:
            offi, dti = fdict['intensities'].offset, fdict['intensities'].datatype
        else:
            offi, dti = None, None

        endian = '>' if msg.is_bigendian else '<'
        fx, fy, fz = endian + PF_FMT[dtx], endian + PF_FMT[dty], endian + PF_FMT[dtz]
        fi = (endian + PF_FMT[dti]) if has_i else None
        data = memoryview(msg.data)
        step = msg.point_step
        n = len(data) // step if step > 0 else 0
        out = np.empty((n, 4), dtype=np.float32)
        for idx in range(n):
            base = idx * step
            x = struct.unpack_from(fx, data, base + offx)[0]
            y = struct.unpack_from(fy, data, base + offy)[0]
            z = struct.unpack_from(fz, data, base + offz)[0]
            inten = float(struct.unpack_from(fi, data, base + offi)[0]) if has_i else 1.0
            out[idx] = (x, y, z, inten)
        return out
    except Exception as e:
        print(f"[LIDAR] fallback 解析失败: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def _write_pcd(arr_xyz_i: np.ndarray, out_path: str, binary: bool = True):
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if arr_xyz_i is None:
        arr_xyz_i = np.zeros((0, 4), dtype=np.float32)
    arr = arr_xyz_i.astype(np.float32, copy=False)
    n = arr.shape[0] if arr.ndim == 2 else 0

    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS x y z intensity",
        "SIZE 4 4 4 4",
        "TYPE F F F F",
        "COUNT 1 1 1 1",
        f"WIDTH {n}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n}",
        "DATA binary" if binary else "DATA ascii"
    ]
    with open(out_path, 'wb') as f:
        f.write(("\n".join(header_lines) + "\n").encode('ascii'))
        if n > 0:
            if binary:
                f.write(arr.tobytes(order='C'))
            else:
                # ascii
                for i in range(n):
                    x, y, z, intensity = arr[i]
                    line = f"{x} {y} {z} {intensity}\n"
                    f.write(line.encode('ascii'))
    return True


def save_pc2_to_pcd(msg, out_path: str, binary: bool = True):
    try:
        arr = _pc2_to_numpy_xyz_i(msg)
        return _write_pcd(arr, out_path, binary=binary)
    except Exception as e:
        print(f"[LIDAR] 保存PCD失败: {e}")
        return False


def write_empty_pcd(out_path: str):
    try:
        return _write_pcd(np.zeros((0, 4), dtype=np.float32), out_path, binary=True)
    except Exception as e:
        print(f"[LIDAR] 写入空PCD失败: {e}")
        return False


def write_radar_csv(rows, out_csv_path: str):
    # rows: list of dicts with expected keys already mapped
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    header = [
        "frame", "id",
        "dist_long", "dist_lat",
        "vrel_long", "vrel_lat",
        "rcs", "dyn_prop",
        "arel_long", "tag", "arel_lat",
        "orientation_angle", "length", "width",
        "class"
    ]
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, delimiter=' ')
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------- main -----------
def parse_and_export(
    bag_path: str,
    lidar_topic: str,
    radar_topic: str,
    out_dir: str,
    sync_by: str = 'bag',
    max_assoc_dt: float = 0.10,
    max_radar_frames: int = None,
    max_lidar_frames: int = None
):
    print(f"[INFO] bag={bag_path}")
    bag = rosbag.Bag(bag_path)
    meta = bag.get_type_and_topic_info().topics

    def info_topic(t):
        if t in meta:
            print(f"[INFO] {t:30s} type={meta[t].msg_type:35s} count={meta[t].message_count}")
        else:
            print(f"[WARN] {t} not found")

    info_topic(lidar_topic)
    info_topic(radar_topic)

    os.makedirs(out_dir, exist_ok=True)

    # 1) 预扫 LiDAR 时间戳列表（用于与雷达帧匹配）
    lid_ts_sorted = []
    lid_cnt_seen = 0
    print("[LIDAR] 预扫描 LiDAR 时间戳...")
    for _, msg, t in bag.read_messages(topics=[lidar_topic]):
        if (max_lidar_frames is not None) and (lid_cnt_seen >= max_lidar_frames):
            break
        ts = t.to_sec() if sync_by == 'bag' else msg.header.stamp.to_sec()
        lid_ts_sorted.append(ts)
        lid_cnt_seen += 1
        if lid_cnt_seen <= 2:
            print(f"[LIDAR] ts={ts:.6f}")
    lid_ts_sorted.sort()
    print(f"[LIDAR] 时间戳数量: {len(lid_ts_sorted)}")

    # 2) 扫描雷达，分帧，并与LiDAR最近邻匹配；逐帧保存CSV
    bag_radar = rosbag.Bag(bag_path)
    frame_idx = -1
    curr_stamp = None
    rows_current_frame = []
    radar_msgs = 0
    assoc_fail = 0
    # 记录每一帧匹配到的lidar ts（None 表示未匹配）
    frame_to_lidar_ts = {}

    def finalize_frame_write_csv(idx, rows):
        out_csv = os.path.join(out_dir, f"{idx}.csv")
        write_radar_csv(rows, out_csv)
        print(f"[RADAR] 写入 {out_csv} targets={len(rows)}")

    print("[RADAR] 遍历并按帧写CSV...")
    for _, msg, t in bag_radar.read_messages(topics=[radar_topic]):
        radar_msgs += 1
        # 基于 header.stamp 分帧
        hdr = getattr(msg, 'header', None)
        if hdr is None:
            # 无header的情况：每条消息视作新帧
            is_new_frame = True
        else:
            is_new_frame = (curr_stamp is None) or (hdr.stamp != curr_stamp)

        if is_new_frame:
            # 写出上一帧
            if frame_idx >= 0:
                finalize_frame_write_csv(frame_idx, rows_current_frame)

            # 帧数限制检查
            if (max_radar_frames is not None) and (frame_idx + 1 >= max_radar_frames):
                print("[RADAR] 达到最大帧数限制，停止读取雷达")
                break

            # 新的一帧
            frame_idx += 1
            curr_stamp = hdr.stamp if hdr is not None else None
            rows_current_frame = []

            # 对这一帧计算与LiDAR的最近邻匹配
            rts = t.to_sec() if (sync_by == 'bag' or hdr is None) else hdr.stamp.to_sec()
            best_ts, best_dt = nearest_ts(lid_ts_sorted, rts)
            if (best_ts is None) or (best_dt > max_assoc_dt):
                frame_to_lidar_ts[frame_idx] = None
                assoc_fail += 1
            else:
                frame_to_lidar_ts[frame_idx] = best_ts

        # 追加当前消息的目标列表到当前帧
        track_list = []
        if hasattr(msg, 'trackList'):
            track_list = msg.trackList
        elif hasattr(msg, 'tracks'):
            track_list = msg.tracks

        for tr in track_list:
            # 安全获取各字段
            dist_long = float(getattr(tr, 'x', 0.0))
            dist_lat = float(getattr(tr, 'y', 0.0))
            vrel_long = float(getattr(tr, 'vx', 0.0))
            vrel_lat = float(getattr(tr, 'vy', 0.0))
            rcs = float(getattr(tr, 'rcs', 0.0))
            # dyn_prop: 优先 trackType 或 dynProp
            dyn_prop = int(getattr(tr, 'trackType', getattr(tr, 'dynProp', 0)))
            arel_long = float(getattr(tr, 'ax', 0.0))
            arel_lat = float(getattr(tr, 'ay', 0.0))
            tag = int(getattr(tr, 'trackState', 0))
            orientation_angle = float(getattr(tr, 'orientation', 0.0))
            length = float(getattr(tr, 'length', 0.0))
            width = float(getattr(tr, 'width', 0.0))
            _id = int(getattr(tr, 'id', 0))

            rows_current_frame.append({
                "frame": frame_idx,
                "id": _id,
                "dist_long": dist_long,
                "dist_lat": dist_lat,
                "vrel_long": vrel_long,
                "vrel_lat": vrel_lat,
                "rcs": rcs,
                "dyn_prop": dyn_prop,
                "arel_long": arel_long,
                "tag": tag,
                "arel_lat": arel_lat,
                "orientation_angle": orientation_angle,
                "length": length,
                "width": width,
                "class": 0  # 默认0
            })

    # 循环结束后，写出最后一帧
    if frame_idx >= 0 and rows_current_frame is not None:
        finalize_frame_write_csv(frame_idx, rows_current_frame)

    bag_radar.close()
    print(f"[RADAR] msgs={radar_msgs}, frames={frame_idx + 1}, 未匹配到LiDAR的帧={assoc_fail}")

    # 3) 根据匹配关系写出每一帧的 LiDAR PCD（按帧号命名）
    #    多个雷达帧可能匹配到同一LiDAR时间戳 -> 对应重复写出多个同内容PCD到不同编号
    ts_to_frames = {}
    total_frames = frame_idx + 1
    for fidx in range(total_frames):
        ts = frame_to_lidar_ts.get(fidx, None)
        if ts is None:
            continue
        ts_to_frames.setdefault(ts, []).append(fidx)

    pcd_written = set()
    # 第二次遍历 LiDAR 消息，找到被匹配到的时间戳并写出对应的PCD
    print("[LIDAR] 第二次遍历，输出PCD文件...")
    bag_lidar = rosbag.Bag(bag_path)
    for _, msg, t in bag_lidar.read_messages(topics=[lidar_topic]):
        ts = t.to_sec() if sync_by == 'bag' else msg.header.stamp.to_sec()
        if ts in ts_to_frames:
            for fidx in ts_to_frames[ts]:
                out_pcd = os.path.join(out_dir, f"{fidx}.pcd")
                # 避免重复写相同帧号
                if fidx in pcd_written:
                    continue
                ok = save_pc2_to_pcd(msg, out_pcd, binary=True)
                if ok:
                    pcd_written.add(fidx)
                    print(f"[LIDAR] 写入 {out_pcd}")
    bag_lidar.close()

    # 为未匹配到LiDAR的帧写入空PCD
    missing_pcd = 0
    for fidx in range(total_frames):
        if fidx not in pcd_written:
            out_pcd = os.path.join(out_dir, f"{fidx}.pcd")
            write_empty_pcd(out_pcd)
            missing_pcd += 1
    if missing_pcd > 0:
        print(f"[LIDAR] 为 {missing_pcd} 帧写入了空PCD文件")

    print(f"[DONE] 输出目录: {out_dir} | 总帧数: {total_frames} | CSV/PCD 按帧命名为 0.csv/0.pcd, 1.csv/1.pcd, ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True, help='/mnt/hgfs/J-2/work/1014data/wj/2025-10-16-10-43-21.bag ')
    ap.add_argument('--radar_topic', default='/radar_track_2')
    ap.add_argument('--lidar_topic', default='/hwPtClound_B1')
    ap.add_argument('--out_dir', default='/mnt/hgfs/J-2/work/1014data/wj/2025-10-16-10-43-21/frames/', help='输出目录，内含 N.csv 和 N.pcd')
    ap.add_argument('--sync_by', choices=['bag', 'header'], default='bag', help='同步使用bag时间或消息header时间')
    ap.add_argument('--max_assoc_dt', type=float, default=0.10, help='雷达与LiDAR时间匹配的最大间隔(s)')
    ap.add_argument('--max_radar_frames', type=int, default=None, help='最多输出多少个雷达帧（从0开始计数）')
    ap.add_argument('--max_lidar_frames', type=int, default=None, help='LiDAR时间戳预扫描的上限（None为不限制）')
    args = ap.parse_args()

    parse_and_export(
        bag_path=args.bag,
        lidar_topic=args.lidar_topic,
        radar_topic=args.radar_topic,
        out_dir=args.out_dir,
        sync_by=args.sync_by,
        max_assoc_dt=args.max_assoc_dt,
        max_radar_frames=args.max_radar_frames,
        max_lidar_frames=args.max_lidar_frames
    )
