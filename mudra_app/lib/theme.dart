import 'package:flutter/material.dart';

/// Change this to rename the app on screen.
const String kAppTitle = 'Hasta Mudras';

/// Android's built-in serif family.
const String kSerif = 'serif';

class Pal {
  static const Color maroon = Color(0xFF6E1423);
  static const Color deepMaroon = Color(0xFF3F0B15);
  static const Color gold = Color(0xFFC9A24B);
  static const Color paleGold = Color(0xFFEAD7A4);
  static const Color parchment = Color(0xFFF4E8CC);
  static const Color card = Color(0xFFFFF9EA);
  static const Color imageBg = Color(0xFFEFE0BC);
  static const Color ink = Color(0xFF2A150F);
  static const Color inkSoft = Color(0xB32A150F); // ink at 70%
  static const Color scrim = Color(0xD93F0B15); // deep maroon at 85%
}

TextStyle display(double size, {Color color = Pal.ink}) => TextStyle(
      fontFamily: kSerif,
      fontSize: size,
      fontWeight: FontWeight.w600,
      color: color,
      height: 1.15,
    );

ThemeData buildTheme() {
  final scheme = ColorScheme.fromSeed(seedColor: Pal.maroon).copyWith(
    primary: Pal.maroon,
    onPrimary: Pal.paleGold,
    secondary: Pal.gold,
    surface: Pal.parchment,
    onSurface: Pal.ink,
  );
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    scaffoldBackgroundColor: Pal.parchment,
  );
}
