import 'package:flutter/material.dart';
import 'home_screen.dart';
import 'mudra_info.dart';
import 'theme.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final info = await loadMudraInfo();
  runApp(MaterialApp(
    title: kAppTitle,
    debugShowCheckedModeBanner: false,
    theme: buildTheme(),
    home: HomeScreen(info: info),
  ));
}
