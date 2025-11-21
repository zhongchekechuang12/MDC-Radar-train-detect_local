#!/usr/bin/env python3
import os, argparse, subprocess, struct
import numpy as np
import pandas as pd
import rosbag
import sensor_msgs.point_cloud2 as pc2
from bisect import bisect_left

# ----------- helpers -----------
PF_FMT = {1:'b',2:'B',3:'h',4:'H',5:'i',6:'I',7:'f',8:'d'}

def nearest(ts_list, ts2id, t_query):
    if not ts_list: return 0, float('inf'), None
    i = bisect_left(ts_list, t_query)
    cand = []
    if i < len(ts_list): cand.append(ts_list[i])
    if i > 0: cand.append(ts_list[i-1])
    best_id, best_dt, best_ts = 0, float('inf'), None
    for ts in cand:
        dt = abs(ts - t_query)
        if dt < best_dt:
            best_dt = dt; best_id = ts2id[ts]; best_ts = ts
    return best_id, best_dt, best_ts

def save_pc2_to_bin(msg, out_path: str):
    # fast path
    try:
        has_i = any(getattr(f,'name','').lower() in ('intensity','intensities') for f in msg.fields)
        fields = ['x','y','z'] + (['intensity'] if has_i else [])
        pts = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
        if len(pts)==0:
            arr = np.zeros((0,4), dtype=np.float32)
        else:
            arr = np.array(pts, dtype=np.float32).reshape(-1, len(fields))
            if arr.shape[1]==3:
                inten = np.ones((arr.shape[0],1), dtype=np.float32)
                arr = np.concatenate([arr,inten], axis=1)
        arr.astype(np.float32).tofile(out_path)
        return True
    except Exception as e:
        print(f"[LIDAR] read_points failed, fallback: {e}")

    # fallback parse
    try:
        fdict = {f.name:f for f in msg.fields}
        for k in ('x','y','z'):
            if k not in fdict:
                print(f"[LIDAR] missing field {k}, skip"); return False
        offx, offy, offz = fdict['x'].offset, fdict['y'].offset, fdict['z'].offset
        dtx, dty, dtz = fdict['x'].datatype, fdict['y'].datatype, fdict['z'].datatype
        has_i = ('intensity' in fdict) or ('intensities' in fdict)
        if 'intensity' in fdict:
            offi, dti = fdict['intensity'].offset, fdict['intensity'].datatype
        elif 'intensities' in fdict:
            offi, dti = fdict['intensities'].offset, fdict['intensities'].datatype
        endian = '>' if msg.is_bigendian else '<'
        fx, fy, fz = endian+PF_FMT[dtx], endian+PF_FMT[dty], endian+PF_FMT[dtz]
        fi = (endian+PF_FMT[dti]) if has_i else None
        data = memoryview(msg.data); step = msg.point_step
        n = len(data)//step if step>0 else 0
        out = np.empty((n,4), dtype=np.float32)
        for idx in range(n):
            base = idx*step
            x = struct.unpack_from(fx, data, base+offx)[0]
            y = struct.unpack_from(fy, data, base+offy)[0]
            z = struct.unpack_from(fz, data, base+offz)[0]
            inten = float(struct.unpack_from(fi, data, base+offi)[0]) if has_i else 1.0
            out[idx] = (x,y,z,inten)
        out.tofile(out_path)
        return True
    except Exception as e:
        print(f"[LIDAR] fallback parse failed: {e}")
        return False

# ----------- main -----------
def parse_and_export(bag_path, out_csv,
                     lidar_topic, lidar_out_dir,
                     radar_topic,
                     sync_by='bag', max_assoc_dt=0.10,
                     max_radar_frames=None, max_lidar_frames=None):
    print(f"[INFO] bag={bag_path}")
    bag = rosbag.Bag(bag_path)
    meta = bag.get_type_and_topic_info().topics
    def info_topic(t):
        if t in meta: print(f"[INFO] {t:30s} type={meta[t].msg_type:35s} count={meta[t].message_count}")
        else:        print(f"[WARN] {t} not found")
    info_topic(lidar_topic); info_topic(radar_topic)

    # 1) export LiDAR (limit frames)
    os.makedirs(lidar_out_dir, exist_ok=True)
    lid_ts2id, lid_ts_sorted = {}, []
    lid_cnt = 0
    for _, msg, t in bag.read_messages(topics=[lidar_topic]):
        if max_lidar_frames and lid_cnt >= max_lidar_frames: break
        ts = t.to_sec() if sync_by=='bag' else msg.header.stamp.to_sec()
        out_bin = os.path.join(lidar_out_dir, f"frame_{lid_cnt+1:06d}.bin")
        if save_pc2_to_bin(msg, out_bin):
            lid_cnt += 1
            lid_ts2id[ts] = lid_cnt
            lid_ts_sorted.append(ts)
            if lid_cnt<=2: print(f"[LIDAR] #{lid_cnt} ts={ts:.6f} -> {out_bin}")
    lid_ts_sorted.sort()
    print(f"[LIDAR] total_saved: {lid_cnt}")

    # 2) 跳过相机数据处理
    print(f"[CAM] 已屏蔽相机数据处理")

    # 3) 跳过相机流解码
    print(f"[CAM] 已屏蔽相机流解码，不生成JPG文件")

    # 4) 处理雷达数据 + 与激光雷达的时间同步
    bag = rosbag.Bag(bag_path)
    rows = []
    radar_frame = 0
    curr_stamp = None
    assoc_l0 = 0  # 激光雷达同步失败计数
    radar_msgs = 0
    for _, msg, t in bag.read_messages(topics=[radar_topic]):
        radar_msgs += 1
        # 新帧开始时检查是否达到上限
        if curr_stamp is None or msg.header.stamp != curr_stamp:
            if max_radar_frames and radar_frame >= max_radar_frames: break
            curr_stamp = msg.header.stamp
            radar_frame += 1

        rts = t.to_sec() if sync_by=='bag' else msg.header.stamp.to_sec()
        # 仅计算与激光雷达的同步
        lid_id, lid_dt, _ = nearest(lid_ts_sorted, lid_ts2id, rts)
        if lid_dt > max_assoc_dt: lid_id = 0; assoc_l0 += 1

        hdr = msg.header
        for tr in msg.trackList:
            rows.append({
                "id": tr.id,
                "x": tr.x, "y": tr.y,
                "vx": tr.vx, "vy": tr.vy,
                "ax": getattr(tr,'ax',0.0), "ay": getattr(tr,'ay',0.0),
                "rcs": tr.rcs,
                "length": getattr(tr,'length',0.0), "width": getattr(tr,'width',0.0),
                "orientation": getattr(tr,'orientation',0.0),
                "trackType": getattr(tr,'trackType',0),
                "existProbability": getattr(tr,'existProbability',0.0),
                "radar_frame": radar_frame,
                "closest_lidar_frame": lid_id,
                "time_diff_with_lidar": (lid_dt if lid_id>0 else None),
                "timestamp_sec": hdr.stamp.secs, "timestamp_nsec": hdr.stamp.nsecs,
                "header_frame_id": hdr.frame_id,
                "trackState": getattr(tr,'trackState',0),
            })
    bag.close()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[RADAR] msgs={radar_msgs}, frames_written={radar_frame}, 激光雷达同步失败={assoc_l0}")
    print(f"[DONE] CSV: {out_csv} rows={len(df)}")
    if not df.empty:
        print(df[['radar_frame','closest_lidar_frame']].drop_duplicates().head(20))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--radar_topic', default='/radar_track_1')
    ap.add_argument('--lidar_topic', default='/hwPtClound_A3')
    ap.add_argument('--out_csv',     default='radar.csv')
    ap.add_argument('--lidar_out_dir', default='lidar_frames')
    ap.add_argument('--sync_by', choices=['bag','header'], default='bag')
    ap.add_argument('--max_assoc_dt', type=float, default=0.10)
    # 限制帧数参数
    ap.add_argument('--max_radar_frames', type=int, default=None)
    ap.add_argument('--max_lidar_frames', type=int, default=None)
    args = ap.parse_args()

    parse_and_export(args.bag, args.out_csv,
                     args.lidar_topic, args.lidar_out_dir,
                     args.radar_topic,
                     sync_by=args.sync_by, max_assoc_dt=args.max_assoc_dt,
                     max_radar_frames=args.max_radar_frames,
                     max_lidar_frames=args.max_lidar_frames)

