import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'mudra_info.dart';
import 'mudra_sheet.dart';
import 'practice_screen.dart';
import 'theme.dart';
import 'widgets.dart';

class LearnScreen extends StatefulWidget {
  final Map<String, dynamic> info;
  const LearnScreen({super.key, required this.info});

  @override
  State<LearnScreen> createState() => _LearnScreenState();
}

class _LearnScreenState extends State<LearnScreen> {
  String _q = '';
  late final List<String> _keys = mudraKeys(widget.info);

  OutlineInputBorder _border(Color c, double w) => OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: c, width: w),
      );

  Widget _card(BuildContext context, String key) {
    final m = widget.info[key] as Map<String, dynamic>;
    return Material(
      color: Pal.card,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
        side: const BorderSide(color: Pal.gold, width: 0.9),
      ),
      child: InkWell(
        onTap: () => showMudraSheet(
          context,
          key,
          widget.info,
          onPractice: () {
            Navigator.of(context).push(
              MaterialPageRoute<void>(
                  builder: (_) => PracticeScreen(initialTarget: key)),
            );
          },
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: Container(
                color: Pal.imageBg,
                child: MudraImage(m['image'] as String?),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 12, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(m['display'] as String,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: display(16)),
                  const SizedBox(height: 2),
                  Text(m['meaning'] as String,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                          color: Pal.inkSoft, fontSize: 13)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final q = _q.trim().toLowerCase();
    final shown = _keys.where((k) {
      final m = widget.info[k] as Map<String, dynamic>;
      final hay = '${m['display']} ${m['meaning']}'.toLowerCase();
      return q.isEmpty || hay.contains(q);
    }).toList();

    return Scaffold(
      backgroundColor: Pal.parchment,
      appBar: AppBar(
        backgroundColor: Pal.maroon,
        foregroundColor: Pal.paleGold,
        systemOverlayStyle: SystemUiOverlayStyle.light,
        title: Text('Learn', style: display(22, color: Pal.paleGold)),
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(18),
          child: TempleBorder(),
        ),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
            child: TextField(
              onChanged: (v) => setState(() => _q = v),
              decoration: InputDecoration(
                hintText: 'Search by name or meaning',
                prefixIcon: const Icon(Icons.search, color: Pal.maroon),
                filled: true,
                fillColor: Pal.card,
                contentPadding: EdgeInsets.zero,
                border: _border(Pal.gold, 1),
                enabledBorder: _border(Pal.gold, 1),
                focusedBorder: _border(Pal.maroon, 1.6),
              ),
            ),
          ),
          Expanded(
            child: shown.isEmpty
                ? Center(
                    child: Text(
                      'No mudra matches "$_q". Try another name or meaning.',
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Pal.inkSoft, fontSize: 15),
                    ),
                  )
                : GridView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisSpacing: 12,
                      crossAxisSpacing: 12,
                      childAspectRatio: 0.74,
                    ),
                    itemCount: shown.length,
                    itemBuilder: (context, i) => _card(context, shown[i]),
                  ),
          ),
        ],
      ),
    );
  }
}
