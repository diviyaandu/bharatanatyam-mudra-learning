import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'theme.dart';

/// A gold strip of diamonds between two thin lines, like a temple border.
class TempleBorder extends StatelessWidget {
  final double height;
  const TempleBorder({super.key, this.height = 18});

  @override
  Widget build(BuildContext context) => SizedBox(
        height: height,
        width: double.infinity,
        child: CustomPaint(painter: _BorderPainter()),
      );
}

class _BorderPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final line = Paint()
      ..color = Pal.gold
      ..strokeWidth = 1.1
      ..style = PaintingStyle.stroke;
    final fill = Paint()..color = Pal.gold;
    canvas.drawLine(const Offset(0, 1.5), Offset(size.width, 1.5), line);
    canvas.drawLine(Offset(0, size.height - 1.5),
        Offset(size.width, size.height - 1.5), line);
    final y = size.height / 2;
    final r = size.height * 0.27;
    final step = size.height * 1.5;
    for (double x = step / 2; x < size.width; x += step) {
      final path = Path()
        ..moveTo(x, y - r)
        ..lineTo(x + r, y)
        ..lineTo(x, y + r)
        ..lineTo(x - r, y)
        ..close();
      canvas.drawPath(path, fill);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// A line-drawn lotus: two rings of petals inside two circles.
class LotusPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = math.min(size.width, size.height) / 2;
    final stroke = Paint()
      ..color = Pal.gold
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.3;
    final fill = Paint()..color = Pal.gold;
    canvas.drawCircle(c, r * 0.98, stroke);
    canvas.drawCircle(c, r * 0.64, stroke);

    void petals(int n, double dist, double w, double h) {
      for (var i = 0; i < n; i++) {
        canvas.save();
        canvas.translate(c.dx, c.dy);
        canvas.rotate(2 * math.pi * i / n);
        canvas.drawOval(
            Rect.fromCenter(center: Offset(0, -dist), width: w, height: h),
            stroke);
        canvas.restore();
      }
    }

    petals(16, r * 0.81, r * 0.14, r * 0.30);
    petals(8, r * 0.40, r * 0.24, r * 0.46);
    canvas.drawCircle(c, r * 0.07, fill);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Shows assets/images/<file>, or a quiet hand icon if the file is missing.
class MudraImage extends StatelessWidget {
  final String? file;
  final BoxFit fit;
  const MudraImage(this.file, {super.key, this.fit = BoxFit.contain});

  Widget _placeholder() => const Center(
        child: Icon(Icons.back_hand_outlined, color: Pal.gold, size: 40),
      );

  @override
  Widget build(BuildContext context) {
    if (file == null) return _placeholder();
    return Image.asset(
      'assets/images/$file',
      fit: fit,
      errorBuilder: (_, __, ___) => _placeholder(),
    );
  }
}

ButtonStyle goldButton() => ElevatedButton.styleFrom(
      backgroundColor: Pal.gold,
      foregroundColor: Pal.deepMaroon,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
    );

ButtonStyle outlineButton() => OutlinedButton.styleFrom(
      foregroundColor: Pal.maroon,
      side: const BorderSide(color: Pal.maroon, width: 1.3),
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
    );
