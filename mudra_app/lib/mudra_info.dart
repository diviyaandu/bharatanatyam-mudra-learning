import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

Future<Map<String, dynamic>> loadMudraInfo() async {
  final raw = await rootBundle.loadString('assets/mudras.json');
  return jsonDecode(raw) as Map<String, dynamic>;
}

/// Class keys (skipping the "_meta" entry), sorted by display name.
List<String> mudraKeys(Map<String, dynamic> info) {
  final keys = info.keys.where((k) => !k.startsWith('_')).toList();
  keys.sort((a, b) {
    final da = (info[a] as Map<String, dynamic>)['display'] as String;
    final db = (info[b] as Map<String, dynamic>)['display'] as String;
    return da.compareTo(db);
  });
  return keys;
}
