import 'dart:math' as math;

List<double> _sub(List<double> a, List<double> b) =>
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
double _dot(List<double> a, List<double> b) =>
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
double _len(List<double> v) => math.sqrt(_dot(v, v));

double _ang(List<double> a, List<double> b, List<double> c) {
  final u = _sub(a, b), v = _sub(c, b);
  final cs = _dot(u, v) / (_len(u) * _len(v) + 1e-9);
  return math.acos(cs.clamp(-1.0, 1.0)) * 180 / math.pi;
}

/// Port of feats() in mudra_features.py. p: 21 landmarks [x, y, z], pixel units.
Map<String, double> feats(List<List<double>> p) {
  final f = <String, double>{};
  const fingers = {
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
  };
  fingers.forEach((n, j) {
    final m = j[0], pi = j[1], d = j[2], t = j[3];
    f['${n}_mcp'] = _ang(p[0], p[m], p[pi]);
    f['${n}_pip'] = _ang(p[m], p[pi], p[d]);
    f['${n}_dip'] = _ang(p[pi], p[d], p[t]);
  });
  f['thumb_mcp'] = _ang(p[1], p[2], p[3]);
  f['thumb_ip'] = _ang(p[2], p[3], p[4]);
  final s = _len(_sub(p[9], p[0])) + 1e-9;
  const tips = [
    [8, 12, 'idx_mid'], [12, 16, 'mid_ring'], [16, 20, 'ring_pinky'],
    [4, 8, 'th_idx'], [4, 12, 'th_mid'], [4, 16, 'th_ring'],
    [4, 20, 'th_pinky'], [4, 5, 'th_idxbase'], [4, 9, 'th_midbase'],
  ];
  for (final t in tips) {
    f['tip_${t[2]}'] = _len(_sub(p[t[0] as int], p[t[1] as int])) / s;
  }
  const spreads = [
    [5, 8, 9, 12, 'idx_mid'], [9, 12, 13, 16, 'mid_ring'],
    [13, 16, 17, 20, 'ring_pinky'], [2, 4, 5, 8, 'th_idx'],
  ];
  const zero = [0.0, 0.0, 0.0];
  for (final sp in spreads) {
    final a = _sub(p[sp[3] as int], p[sp[2] as int]);
    final b = _sub(p[sp[1] as int], p[sp[0] as int]);
    f['spread_${sp[4]}'] = _ang(b, zero, a);
  }
  final u = _sub(p[5], p[0]), v = _sub(p[17], p[0]);
  final n = [
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
  ];
  final nl = _len(n) + 1e-9;
  f['palm_tilt'] =
      math.acos((n[2] / nl).abs().clamp(0.0, 1.0)) * 180 / math.pi;
  f['palm_roll'] =
      math.atan2(p[9][1] - p[0][1], p[9][0] - p[0][0]) * 180 / math.pi;
  return f;
}