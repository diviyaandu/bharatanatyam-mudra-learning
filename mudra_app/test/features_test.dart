import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:mudra_app/features.dart';

void main() {
  test('Dart features match Python', () {
    final cases =
        jsonDecode(File('assets/feat_cases.json').readAsStringSync()) as List;
    for (final c in cases) {
      final p = [
        for (final r in c['landmarks'] as List)
          [for (final v in r as List) (v as num).toDouble()]
      ];
      final want = (c['feats'] as Map).map((k, v) => MapEntry(k as String, (v as num).toDouble()));
      final got = feats(p);
      expect(got.length, want.length);
      var worst = 0.0;
      String worstKey = '';
      want.forEach((k, v) {
        expect(got.containsKey(k), true, reason: 'missing $k');
        final d = (got[k]! - v).abs();
        if (d > worst) {
          worst = d;
          worstKey = k;
        }
      });
      // ignore: avoid_print
      print('${c['label']}: worst difference $worst ($worstKey)');
      expect(worst, lessThan(1e-4));
    }
  });
}