import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'learn_screen.dart';
import 'practice_screen.dart';
import 'theme.dart';
import 'widgets.dart';

class HomeScreen extends StatelessWidget {
  final Map<String, dynamic> info;
  const HomeScreen({super.key, required this.info});

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: Pal.deepMaroon,
        body: SafeArea(
          child: Column(
            children: [
              const TempleBorder(),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    children: [
                      const SizedBox(height: 28),
                      SizedBox(
                        width: 200,
                        height: 200,
                        child: CustomPaint(painter: LotusPainter()),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        kAppTitle,
                        textAlign: TextAlign.center,
                        style: display(36, color: Pal.paleGold),
                      ),
                      const SizedBox(height: 10),
                      const Text(
                        'Learn the single-hand gestures of Bharatanatyam, '
                        'then practice them in front of your camera.',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            color: Pal.paleGold, fontSize: 15, height: 1.45),
                      ),
                      const SizedBox(height: 32),
                      _HomeTile(
                        icon: Icons.menu_book,
                        title: 'Learn',
                        subtitle:
                            'See each mudra with a photo, its meaning, and how to form it.',
                        filled: true,
                        onTap: () => Navigator.of(context).push(
                          MaterialPageRoute<void>(
                              builder: (_) => LearnScreen(info: info)),
                        ),
                      ),
                      const SizedBox(height: 14),
                      _HomeTile(
                        icon: Icons.photo_camera,
                        title: 'Practice',
                        subtitle:
                            'Hold a mudra in front of the camera and get a score.',
                        filled: false,
                        onTap: () => Navigator.of(context).push(
                          MaterialPageRoute<void>(
                              builder: (_) => const PracticeScreen()),
                        ),
                      ),
                      const SizedBox(height: 24),
                    ],
                  ),
                ),
              ),
              const TempleBorder(),
            ],
          ),
        ),
      ),
    );
  }
}

class _HomeTile extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool filled;
  final VoidCallback onTap;

  const _HomeTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.filled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final fg = filled ? Pal.deepMaroon : Pal.paleGold;
    return Material(
      color: filled ? Pal.gold : Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: const BorderSide(color: Pal.gold, width: 1.4),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
          child: Row(
            children: [
              Icon(icon, color: fg, size: 30),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: display(22, color: fg)),
                    const SizedBox(height: 2),
                    Text(subtitle,
                        style: TextStyle(color: fg, fontSize: 14, height: 1.3)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
