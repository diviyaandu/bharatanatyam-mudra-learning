import numpy as np
def norm(p):
    p = p - p[0]; v = p[9, :2]; t = -np.pi/2 - np.arctan2(v[1], v[0])
    c, s = np.cos(t), np.sin(t); q = p.copy()
    q[:,0] = p[:,0]*c - p[:,1]*s; q[:,1] = p[:,0]*s + p[:,1]*c
    return (q / (np.linalg.norm(v) + 1e-9)).ravel()

def ang(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u)*np.linalg.norm(v) + 1e-9), -1, 1)))

def feats(p):
    f = {}
    for n, (m, pi, d, t) in {"index": (5,6,7,8), "middle": (9,10,11,12),
                             "ring": (13,14,15,16), "pinky": (17,18,19,20)}.items():
        f[f"{n}_mcp"] = ang(p[0], p[m], p[pi])
        f[f"{n}_pip"] = ang(p[m], p[pi], p[d])
        f[f"{n}_dip"] = ang(p[pi], p[d], p[t])
    f["thumb_mcp"] = ang(p[1], p[2], p[3]); f["thumb_ip"] = ang(p[2], p[3], p[4])
    s = np.linalg.norm(p[9] - p[0]) + 1e-9
    for a, b, nm in [(8,12,"idx_mid"),(12,16,"mid_ring"),(16,20,"ring_pinky"),
                     (4,8,"th_idx"),(4,12,"th_mid"),(4,16,"th_ring"),(4,20,"th_pinky"),
                     (4,5,"th_idxbase"),(4,9,"th_midbase")]:
        f[f"tip_{nm}"] = np.linalg.norm(p[a] - p[b]) / s
    for (m1,t1),(m2,t2),nm in [((5,8),(9,12),"idx_mid"),((9,12),(13,16),"mid_ring"),
                               ((13,16),(17,20),"ring_pinky"),((2,4),(5,8),"th_idx")]:
        f[f"spread_{nm}"] = ang(p[t1]-p[m1], np.zeros(3), p[t2]-p[m2])
    n_ = np.cross(p[5]-p[0], p[17]-p[0]); n_ /= np.linalg.norm(n_) + 1e-9
    f["palm_tilt"] = np.degrees(np.arccos(abs(n_[2])))
    f["palm_roll"] = np.degrees(np.arctan2(p[9,1]-p[0,1], p[9,0]-p[0,0]))
    return f
