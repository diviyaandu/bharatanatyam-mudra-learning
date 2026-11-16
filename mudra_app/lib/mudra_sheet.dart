import 'package:flutter/material.dart';
import 'theme.dart';
import 'widgets.dart';

Widget _section(String title, String body) => Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: display(18, color: Pal.maroon)),
        const SizedBox(height: 4),
        Text(body,
            style:
                const TextStyle(fontSize: 16, height: 1.5, color: Pal.ink)),
        const SizedBox(height: 16),
      ],
    );

/// Bottom sheet with the picture, meaning, how to form it, and traditional uses.
/// Pass [onPractice] to add a "Practice this mudra" button.
Future<void> showMudraSheet(
  BuildContext context,
  String key,
  Map<String, dynamic> info, {
  VoidCallback? onPractice,
}) {
  final m = info[key] as Map<String, dynamic>?;
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: Pal.parchment,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
    ),
    builder: (ctx) => SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(22, 14, 22, 22),
        child: m == null
            ? Text('No description yet for $key.')
            : Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Pal.gold,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  Text(m['display'] as String,
                      style: display(30, color: Pal.maroon)),
                  Text(
                    m['meaning'] as String,
                    style: const TextStyle(
                        fontSize: 16,
                        fontStyle: FontStyle.italic,
                        color: Pal.inkSoft),
                  ),
                  const SizedBox(height: 14),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Container(
                      height: 240,
                      width: double.infinity,
                      color: Pal.imageBg,
                      child: MudraImage(m['image'] as String?),
                    ),
                  ),
                  const SizedBox(height: 18),
                  _section('How to form it', m['form'] as String),
                  _section('Traditionally used for', m['uses'] as String),
                  if (m['note'] != null)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: Text(
                        m['note'] as String,
                        style: const TextStyle(
                            fontSize: 14,
                            fontStyle: FontStyle.italic,
                            color: Pal.maroon),
                      ),
                    ),
                  if (onPractice != null)
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: goldButton(),
                        onPressed: () {
                          Navigator.of(ctx).pop();
                          onPractice();
                        },
                        child: const Text('Practice this mudra'),
                      ),
                    ),
                ],
              ),
      ),
    ),
  );
}
