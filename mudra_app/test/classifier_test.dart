import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:mudra_app/classifier.dart';

void main() {
  test('Dart output matches sklearn on exported cases', () {
    final clf =
        MudraClassifier.fromJson(File('assets/mlp.json').readAsStringSync());
    final cases =
        jsonDecode(File('assets/test_cases.json').readAsStringSync()) as List;
    for (final c in cases) {
      final p = [
        for (final r in c['landmarks'] as List)
          [for (final v in r as List) (v as num).toDouble()]
      ];
      final want = [for (final v in c['proba'] as List) (v as num).toDouble()];
      final got = clf.predictProba(p);
      var worst = 0.0;
      for (var i = 0; i < want.length; i++) {
        worst = math.max(worst, (got[i] - want[i]).abs());
      }
      // ignore: avoid_print
      print('${c['label']}: worst difference $worst');
      expect(worst, lessThan(1e-6));
    }
  });
}