import 'dart:convert';
import 'dart:math' as math;

class MudraClassifier {
  final List<String> classes;
  final List<double> mean, scale;
  final List<List<List<double>>> weights;
  final List<List<double>> biases;

  MudraClassifier._(
      this.classes, this.mean, this.scale, this.weights, this.biases);

  factory MudraClassifier.fromJson(String src) {
    final m = jsonDecode(src) as Map<String, dynamic>;
    List<double> vec(dynamic l) =>
        (l as List).map((e) => (e as num).toDouble()).toList();
    return MudraClassifier._(
      (m['classes'] as List).cast<String>(),
      vec(m['mean']),
      vec(m['scale']),
      (m['weights'] as List).map((w) => (w as List).map(vec).toList()).toList(),
      (m['biases'] as List).map(vec).toList(),
    );
  }

  /// Port of norm() from mudra_features.py.
  /// p: 21 landmarks, each [x, y, z], in pixel units.
  static List<double> normalize(List<List<double>> p) {
    final o = p[0];
    final q = [
      for (final r in p) [r[0] - o[0], r[1] - o[1], r[2] - o[2]]
    ];
    final vx = q[9][0], vy = q[9][1];
    final t = -math.pi / 2 - math.atan2(vy, vx);
    final c = math.cos(t), s = math.sin(t);
    final n = math.sqrt(vx * vx + vy * vy) + 1e-9;
    final out = <double>[];
    for (final r in q) {
      out
        ..add((r[0] * c - r[1] * s) / n)
        ..add((r[0] * s + r[1] * c) / n)
        ..add(r[2] / n);
    }
    return out;
  }

  List<double> predictProba(List<List<double>> p) {
    var x = normalize(p);
    for (var i = 0; i < x.length; i++) {
      x[i] = (x[i] - mean[i]) / scale[i];
    }
    for (var l = 0; l < weights.length; l++) {
      final w = weights[l];
      final out = List<double>.from(biases[l]);
      for (var i = 0; i < x.length; i++) {
        final xi = x[i];
        final wi = w[i];
        for (var j = 0; j < out.length; j++) {
          out[j] += xi * wi[j];
        }
      }
      if (l < weights.length - 1) {
        for (var j = 0; j < out.length; j++) {
          if (out[j] < 0) out[j] = 0.0;
        }
      }
      x = out;
    }
    final mx = x.reduce(math.max);
    final e = [for (final v in x) math.exp(v - mx)];
    final sum = e.fold(0.0, (a, b) => a + b);
    return [for (final v in e) v / sum];
  }
}