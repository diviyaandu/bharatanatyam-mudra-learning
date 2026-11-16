import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'classifier.dart';
import 'features.dart';
import 'mudra_info.dart';
import 'mudra_sheet.dart';
import 'theme.dart';
import 'widgets.dart';

const double kTrainSign = -1.0;
const bool kSwapAxes = true;
const bool kInvertX = true;
const bool kInvertY = true;
const double kGoodZ = 3.0; // below this gap, show "looks good"

/// Set to true to show the raw numbers (pred, P, match, z) while testing.
const bool kShowDebug = false;

const Map<String, List<int>> kJoints = {
  'index': [5, 6, 7, 8],
  'idx': [5, 6, 7, 8],
  'idxbase': [5, 6, 7, 8],
  'middle': [9, 10, 11, 12],
  'mid': [9, 10, 11, 12],
  'midbase': [9, 10, 11, 12],
  'ring': [13, 14, 15, 16],
  'pinky': [17, 18, 19, 20],
  'thumb': [1, 2, 3, 4],
  'th': [1, 2, 3, 4],
};

enum Phase { live, holding, result }

class FixInfo {
  final String text;
  final Set<int> hl;
  final double worstAbsZ;
  final String? hint;
  FixInfo(this.text, this.hl, this.worstAbsZ, this.hint);
}

class Analysis {
  final bool hasHand;
  final String pred;
  final double predConf;
  final double pt;
  final String fixLine;
  final String debug;
  final Set<int> hl;
  final double? score;
  final String? hint;
  const Analysis({
    this.hasHand = false,
    this.pred = '',
    this.predConf = 0,
    this.pt = 0,
    this.fixLine = '',
    this.debug = '',
    this.hl = const <int>{},
    this.score,
    this.hint,
  });
}

class DotPainter extends CustomPainter {
  final List<Offset> pts;
  final Set<int> hl;
  DotPainter(this.pts, this.hl);

  @override
  void paint(Canvas canvas, Size size) {
    for (var i = 0; i < pts.length; i++) {
      final red = hl.contains(i);
      canvas.drawCircle(pts[i], red ? 9 : 5,
          Paint()..color = red ? Colors.redAccent : Colors.greenAccent);
    }
  }

  @override
  bool shouldRepaint(DotPainter old) => true;
}

class PracticeScreen extends StatefulWidget {
  final String initialTarget;
  const PracticeScreen({super.key, this.initialTarget = 'Pathaka'});

  @override
  State<PracticeScreen> createState() => _PracticeScreenState();
}

class _PracticeScreenState extends State<PracticeScreen> {
  HandLandmarkerPlugin? _plugin;
  CameraController? _controller;
  StreamSubscription<List<Hand>>? _sub;
  MudraClassifier? _clf;
  Map<String, dynamic>? _stats;
  Map<String, dynamic>? _info;
  String _status = 'Starting the camera...';
  String _target = 'Pathaka';

  List<Landmark> _lms = const [];
  Analysis _a = const Analysis();

  Phase _phase = Phase.live;
  DateTime _holdStart = DateTime.now();
  Timer? _timer;
  final List<double> _scores = [];
  final Map<String, int> _hints = {};
  double? _resultScore;
  String _resultText = '';

  @override
  void initState() {
    super.initState();
    _target = widget.initialTarget;
    _init();
  }

  Future<void> _init() async {
    try {
      _clf = MudraClassifier.fromJson(
          await rootBundle.loadString('assets/mlp.json'));
      _stats = jsonDecode(await rootBundle.loadString('assets/class_stats.json'))
          as Map<String, dynamic>;
      _info = await loadMudraInfo();
      final cams = await availableCameras();
      final cam = cams.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.front,
          orElse: () => cams.first);
      _controller = CameraController(cam, ResolutionPreset.medium,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.yuv420);
      _plugin = HandLandmarkerPlugin.create(
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          delegate: HandLandmarkerDelegate.cpu);
      await _controller!.initialize();
      if (!mounted) return;
      await _controller!.startImageStream((img) {
        _plugin?.processFrame(img, _controller!.description.sensorOrientation);
      });
      _sub = _plugin!.landmarkStream.listen(_onHands);
      if (mounted) setState(() => _status = 'ready');
    } catch (e) {
      if (mounted) setState(() => _status = 'error: $e');
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _sub?.cancel();
    _controller?.stopImageStream();
    _controller?.dispose();
    _plugin?.dispose();
    super.dispose();
  }

  void _onHands(List<Hand> hands) {
    final ps = _controller!.value.previewSize!;
    final a = _analyze(hands, ps);
    if (_phase == Phase.holding && a.hasHand) {
      final ms = DateTime.now().difference(_holdStart).inMilliseconds;
      if (ms >= 1000) {
        _scores.add(a.score!);
        final h = a.hint;
        if (h != null && h.isNotEmpty) _hints[h] = (_hints[h] ?? 0) + 1;
      }
    }
    if (!mounted) return;
    setState(() {
      _lms = hands.isEmpty ? const [] : hands.first.landmarks;
      _a = a;
    });
  }

  void _start() {
    _timer?.cancel();
    _scores.clear();
    _hints.clear();
    setState(() {
      _phase = Phase.holding;
      _holdStart = DateTime.now();
    });
    _timer = Timer(const Duration(seconds: 3), _finish);
  }

  void _finish() {
    if (!mounted) return;
    if (_scores.length < 8) {
      setState(() {
        _phase = Phase.result;
        _resultScore = null;
        _resultText =
            'Your hand was not clearly visible. Hold it 15 to 30 cm from the camera and try again.';
      });
      return;
    }
    final s = [..._scores]..sort();
    final med = s[s.length ~/ 2];
    var tip = 'Nice and steady.';
    if (_hints.isNotEmpty) {
      final top = _hints.entries.reduce((x, y) => x.value >= y.value ? x : y);
      if (top.value >= _scores.length * 0.3) tip = _cap(top.key);
    }
    setState(() {
      _phase = Phase.result;
      _resultScore = med;
      _resultText = tip;
    });
  }

  String _cap(String s) =>
      s.isEmpty ? s : '${s[0].toUpperCase()}${s.substring(1)}.';

  String _grade(double s) =>
      s >= 85 ? 'Excellent' : (s >= 65 ? 'Good' : 'Keep practicing');

  String _name(String k) {
    final m = _info?[k];
    if (m is Map<String, dynamic>) return (m['display'] as String?) ?? k;
    return k;
  }

  Analysis _analyze(List<Hand> hands, Size ps) {
    if (hands.isEmpty) return const Analysis();
    final clf = _clf!;
    var p = [
      for (final l in hands.first.landmarks)
        [l.x * ps.width, l.y * ps.height, l.z * ps.width]
    ];
    final cross = (p[5][0] - p[0][0]) * (p[17][1] - p[0][1]) -
        (p[5][1] - p[0][1]) * (p[17][0] - p[0][0]);
    final mirrored = cross * kTrainSign < 0;
    if (mirrored) {
      p = [
        for (final r in p) [-r[0], r[1], r[2]]
      ];
    }
    final pr = clf.predictProba(p);
    var best = 0;
    for (var i = 1; i < pr.length; i++) {
      if (pr[i] > pr[best]) best = i;
    }
    final pt = pr[clf.classes.indexOf(_target)];
    final fix = _fix(p);
    final featPart =
        (1 - (fix.worstAbsZ - 2.0) / 8.0).clamp(0.0, 1.0).toDouble();
    final score = 100.0 * (0.5 * pt + 0.5 * featPart);
    final String fixLine = fix.hint == null
        ? 'Looks good. Hold steady.'
        : (fix.hint!.isEmpty
            ? 'Adjust the highlighted joints.'
            : _cap(fix.hint!));
    return Analysis(
      hasHand: true,
      pred: clf.classes[best],
      predConf: pr[best],
      pt: pt,
      fixLine: fixLine,
      debug: 'pred ${clf.classes[best]} ${pr[best].toStringAsFixed(2)}\n'
          '$_target P=${pt.toStringAsFixed(2)}  match ${score.round()}\n'
          '${fix.text}',
      hl: fix.hl,
      score: score,
      hint: fix.hint,
    );
  }

  FixInfo _fix(List<List<double>> p) {
    final med = _stats!['median'][_target] as Map<String, dynamic>;
    final sd = _stats!['sd'][_target] as Map<String, dynamic>;
    String? top;
    var topZ = 0.0;
    for (final e in feats(p).entries) {
      if (e.key == 'palm_roll' || !med.containsKey(e.key)) continue;
      final z = (e.value - (med[e.key] as num).toDouble()) /
          (sd[e.key] as num).toDouble();
      if (top == null || z.abs() > topZ.abs()) {
        top = e.key;
        topZ = z;
      }
    }
    if (top == null) return FixInfo('', <int>{}, 0.0, null);
    final az = topZ.abs();
    if (az < kGoodZ) {
      return FixInfo(
          'looks good (${az.toStringAsFixed(1)})', <int>{}, az, null);
    }
    final high = topZ > 0;
    final hl = <int>{};
    for (final t in top.split('_')) {
      hl.addAll(kJoints[t] ?? const <int>[]);
    }
    final hint = _hint(top, high);
    return FixInfo(
        'fix: $top ${high ? 'high' : 'low'} (${topZ.toStringAsFixed(1)})\n$hint',
        hl,
        az,
        hint);
  }

  String _hint(String k, bool high) {
    const joint = {
      'mcp': 'base knuckle',
      'pip': 'middle joint',
      'dip': 'top joint',
      'ip': 'top joint'
    };
    final parts = k.split('_');
    if (parts.length == 2 && joint.containsKey(parts[1])) {
      return '${parts[0]} ${joint[parts[1]]}: ${high ? 'straighten' : 'bend'} more';
    }
    if (k.startsWith('tip_')) {
      return high ? 'bring these closer' : 'move these apart';
    }
    if (k.startsWith('spread_')) {
      return high ? 'bring fingers together' : 'spread fingers apart';
    }
    if (k == 'palm_tilt') {
      return high ? 'face palm toward camera' : 'tilt palm slightly';
    }
    return '';
  }

  // ---------- UI pieces ----------

  Widget _roundButton(IconData icon, VoidCallback onTap, String tip) {
    return Material(
      color: Pal.scrim,
      shape: const CircleBorder(side: BorderSide(color: Pal.gold, width: 0.9)),
      child: IconButton(
        icon: Icon(icon, color: Pal.gold),
        tooltip: tip,
        onPressed: onTap,
      ),
    );
  }

  Widget _topBar(BuildContext context) {
    return Row(
      children: [
        _roundButton(Icons.arrow_back, () => Navigator.of(context).pop(), 'Back'),
        const SizedBox(width: 8),
        Expanded(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14),
            decoration: BoxDecoration(
              color: Pal.scrim,
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: Pal.gold, width: 0.9),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                isExpanded: true,
                value: _target,
                dropdownColor: Pal.deepMaroon,
                iconEnabledColor: Pal.gold,
                style: const TextStyle(color: Pal.paleGold, fontSize: 16),
                items: [
                  for (final n in _clf!.classes)
                    DropdownMenuItem(value: n, child: Text(_name(n))),
                ],
                onChanged: (v) => setState(() {
                  _target = v!;
                  _timer?.cancel();
                  _phase = Phase.live;
                }),
              ),
            ),
          ),
        ),
        const SizedBox(width: 8),
        _roundButton(Icons.menu_book,
            () => showMudraSheet(context, _target, _info!), 'How to form this mudra'),
      ],
    );
  }

  Widget _panel(Analysis a) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Pal.scrim,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Pal.gold, width: 0.9),
      ),
      child: !a.hasHand
          ? const Text('Show your hand to the camera.',
              style: TextStyle(color: Pal.paleGold, fontSize: 15))
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Column(
                      children: [
                        Text('${a.score!.round()}',
                            style: display(40, color: Pal.gold)),
                        const Text('match',
                            style:
                                TextStyle(color: Pal.paleGold, fontSize: 12)),
                      ],
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Looks like ${_name(a.pred)}',
                              style: const TextStyle(
                                  color: Pal.paleGold, fontSize: 15)),
                          const SizedBox(height: 4),
                          Text(a.fixLine,
                              style: const TextStyle(
                                  color: Colors.white, fontSize: 15)),
                        ],
                      ),
                    ),
                  ],
                ),
                if (kShowDebug)
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: Text(a.debug,
                        style: const TextStyle(
                            color: Colors.white70, fontSize: 12)),
                  ),
              ],
            ),
    );
  }

  Widget _banner(String text) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Pal.scrim,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Pal.gold, width: 0.9),
      ),
      child: Text(text,
          textAlign: TextAlign.center,
          style: display(24, color: Pal.paleGold)),
    );
  }

  Widget _resultCard() {
    return Center(
      child: Container(
        margin: const EdgeInsets.all(24),
        padding: const EdgeInsets.all(22),
        decoration: BoxDecoration(
          color: Pal.parchment,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: Pal.gold, width: 1.4),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(_name(_target), style: display(22, color: Pal.maroon)),
            if (_resultScore != null) ...[
              Text('${_resultScore!.round()}',
                  style: display(72, color: Pal.maroon)),
              Text(_grade(_resultScore!),
                  style: const TextStyle(fontSize: 18, color: Pal.ink)),
            ],
            const SizedBox(height: 10),
            Text(_resultText,
                textAlign: TextAlign.center,
                style: const TextStyle(
                    fontSize: 16, height: 1.4, color: Pal.ink)),
            const SizedBox(height: 18),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                OutlinedButton(
                  style: outlineButton(),
                  onPressed: () => setState(() => _phase = Phase.live),
                  child: const Text('Done'),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  style: goldButton(),
                  onPressed: _start,
                  child: const Text('Try again'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    if (c == null ||
        !c.value.isInitialized ||
        _clf == null ||
        _stats == null ||
        _info == null) {
      final failed = _status.startsWith('error');
      return Scaffold(
        backgroundColor: Pal.deepMaroon,
        appBar: AppBar(
          backgroundColor: Pal.deepMaroon,
          foregroundColor: Pal.paleGold,
          systemOverlayStyle: SystemUiOverlayStyle.light,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (!failed)
                  const CircularProgressIndicator(color: Pal.gold),
                const SizedBox(height: 16),
                Text(_status,
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Pal.paleGold, fontSize: 15)),
              ],
            ),
          ),
        ),
      );
    }

    final ps = c.value.previewSize!;
    final up = Size(ps.height, ps.width);
    String? banner;
    if (_phase == Phase.holding) {
      final ms = DateTime.now().difference(_holdStart).inMilliseconds;
      banner = ms < 1000
          ? 'Get ready...'
          : 'Hold steady... ${((3000 - ms) / 1000).ceil()}';
    }

    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: LayoutBuilder(builder: (context, box) {
          final sw = box.maxWidth, sh = box.maxHeight;
          final scale = math.max(sw / up.width, sh / up.height);
          final ox = (sw - up.width * scale) / 2;
          final oy = (sh - up.height * scale) / 2;
          final pts = <Offset>[];
          for (final l in _lms) {
            var fx = kSwapAxes ? l.y : l.x;
            var fy = kSwapAxes ? l.x : l.y;
            if (kInvertX) fx = 1 - fx;
            if (kInvertY) fy = 1 - fy;
            pts.add(Offset(
                ox + fx * up.width * scale, oy + fy * up.height * scale));
          }
          return Stack(
            children: [
              Positioned.fill(
                child: ClipRect(
                  child: FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: 1000 * (1 / c.value.aspectRatio),
                      height: 1000,
                      child: CameraPreview(c),
                    ),
                  ),
                ),
              ),
              Positioned.fill(
                child: CustomPaint(painter: DotPainter(pts, _a.hl)),
              ),
              Positioned.fill(
                child: SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.all(10),
                    child: Column(
                      children: [
                        _topBar(context),
                        const SizedBox(height: 10),
                        _panel(_a),
                        const Spacer(),
                        if (banner != null) _banner(banner),
                        if (_phase == Phase.live)
                          SizedBox(
                            width: double.infinity,
                            child: ElevatedButton(
                              style: goldButton(),
                              onPressed: _start,
                              child: const Text('Start 3-second check'),
                            ),
                          ),
                        const SizedBox(height: 8),
                      ],
                    ),
                  ),
                ),
              ),
              if (_phase == Phase.result) _resultCard(),
            ],
          );
        }),
      ),
    );
  }
}
