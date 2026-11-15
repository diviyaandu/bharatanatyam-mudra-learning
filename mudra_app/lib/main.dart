import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:hand_landmarker/hand_landmarker.dart';
import 'classifier.dart';
import 'features.dart';

const double kTrainSign = -1.0;
const bool kSwapAxes = true;
const bool kInvertX = true;
const bool kInvertY = true;
const double kGoodZ = 3.0; // below this gap, show "looks good"

const Map<String, List<int>> kJoints = {
  'index': [5, 6, 7, 8], 'idx': [5, 6, 7, 8], 'idxbase': [5, 6, 7, 8],
  'middle': [9, 10, 11, 12], 'mid': [9, 10, 11, 12], 'midbase': [9, 10, 11, 12],
  'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20],
  'thumb': [1, 2, 3, 4], 'th': [1, 2, 3, 4],
};

void main() => runApp(const MaterialApp(home: MudraScreen()));

enum Phase { live, holding, result }

class FixInfo {
  final String text;
  final Set<int> hl;
  final double worstAbsZ;
  final String? hint;
  FixInfo(this.text, this.hl, this.worstAbsZ, this.hint);
}

class Analysis {
  final String text;
  final Set<int> hl;
  final double? score; // null when no hand
  final String? hint;
  Analysis(this.text, this.hl, this.score, this.hint);
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
          Paint()..color = red ? Colors.red : Colors.greenAccent);
    }
  }

  @override
  bool shouldRepaint(DotPainter old) => true;
}

class MudraScreen extends StatefulWidget {
  const MudraScreen({super.key});
  @override
  State<MudraScreen> createState() => _MudraScreenState();
}

class _MudraScreenState extends State<MudraScreen> {
  HandLandmarkerPlugin? _plugin;
  CameraController? _controller;
  StreamSubscription<List<Hand>>? _sub;
  MudraClassifier? _clf;
  Map<String, dynamic>? _stats;
  String _status = 'starting...';
  String _target = 'Pathaka';

  List<Landmark> _lms = const [];
  Analysis _a = Analysis('no hand', <int>{}, null, null);

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
    _init();
  }

  Future<void> _init() async {
    try {
      _clf = MudraClassifier.fromJson(await rootBundle.loadString('assets/mlp.json'));
      _stats = jsonDecode(await rootBundle.loadString('assets/class_stats.json'))
          as Map<String, dynamic>;
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
    if (_phase == Phase.holding && a.score != null) {
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
        _resultText = 'Hand not clearly visible. Try again.';
      });
      return;
    }
    final s = [..._scores]..sort();
    final med = s[s.length ~/ 2];
    var tip = 'Nice and steady.';
    if (_hints.isNotEmpty) {
      final top = _hints.entries.reduce((x, y) => x.value >= y.value ? x : y);
      if (top.value >= _scores.length * 0.3) tip = top.key;
    }
    setState(() {
      _phase = Phase.result;
      _resultScore = med;
      _resultText = tip;
    });
  }

  String _grade(double s) =>
      s >= 85 ? 'Excellent' : (s >= 65 ? 'Good' : 'Keep practicing');

  Analysis _analyze(List<Hand> hands, Size ps) {
    if (hands.isEmpty) return Analysis('no hand', <int>{}, null, null);
    final clf = _clf!;
    var p = [
      for (final l in hands.first.landmarks)
        [l.x * ps.width, l.y * ps.height, l.z * ps.width]
    ];
    final cross = (p[5][0] - p[0][0]) * (p[17][1] - p[0][1]) -
        (p[5][1] - p[0][1]) * (p[17][0] - p[0][0]);
    final mirrored = cross * kTrainSign < 0;
    if (mirrored) {
      p = [for (final r in p) [-r[0], r[1], r[2]]];
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
    return Analysis(
        'pred ${clf.classes[best]} ${pr[best].toStringAsFixed(2)}\n'
        '$_target P=${pt.toStringAsFixed(2)}  match ${score.round()}\n'
        '${fix.text}',
        fix.hl,
        score,
        fix.hint);
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
      return FixInfo('looks good (${az.toStringAsFixed(1)})', <int>{}, az, null);
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
    if (k.startsWith('tip_')) return high ? 'bring these closer' : 'move these apart';
    if (k.startsWith('spread_')) {
      return high ? 'bring fingers together' : 'spread fingers apart';
    }
    if (k == 'palm_tilt') return high ? 'face palm toward camera' : 'tilt palm slightly';
    return '';
  }

  @override
  Widget build(BuildContext context) {
    final c = _controller;
    if (c == null || !c.value.isInitialized || _clf == null || _stats == null) {
      return Scaffold(body: Center(child: Text(_status)));
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
    return Scaffold(
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
        return Stack(children: [
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
          Positioned.fill(child: CustomPaint(painter: DotPainter(pts, _a.hl))),
          Positioned(
            left: 8,
            right: 8,
            top: 40,
            child: Container(
                color: Colors.black54,
                padding: const EdgeInsets.all(8),
                child: Text(_a.text,
                    style: const TextStyle(color: Colors.white, fontSize: 16))),
          ),
          if (banner != null)
            Positioned(
              left: 8,
              right: 8,
              bottom: 150,
              child: Container(
                  color: Colors.black87,
                  padding: const EdgeInsets.all(10),
                  child: Text(banner,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 24,
                          fontWeight: FontWeight.bold))),
            ),
          if (_phase == Phase.live)
            Positioned(
              left: 8,
              right: 8,
              bottom: 88,
              child: ElevatedButton(
                  onPressed: _start, child: const Text('Start 3-second check')),
            ),
          Positioned(
            left: 8,
            right: 8,
            bottom: 24,
            child: Container(
              color: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: DropdownButton<String>(
                isExpanded: true,
                value: _target,
                items: [
                  for (final n in _clf!.classes)
                    DropdownMenuItem(value: n, child: Text(n))
                ],
                onChanged: (v) => setState(() {
                  _target = v!;
                  _timer?.cancel();
                  _phase = Phase.live;
                }),
              ),
            ),
          ),
          if (_phase == Phase.result)
            Center(
              child: Card(
                margin: const EdgeInsets.all(24),
                child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(mainAxisSize: MainAxisSize.min, children: [
                    Text(_target, style: const TextStyle(fontSize: 18)),
                    if (_resultScore != null) ...[
                      Text('${_resultScore!.round()}',
                          style: const TextStyle(
                              fontSize: 64, fontWeight: FontWeight.bold)),
                      Text(_grade(_resultScore!),
                          style: const TextStyle(fontSize: 18)),
                    ],
                    const SizedBox(height: 8),
                    Text(_resultText, textAlign: TextAlign.center),
                    const SizedBox(height: 12),
                    ElevatedButton(
                        onPressed: _start, child: const Text('Try again')),
                  ]),
                ),
              ),
            ),
        ]);
      }),
    );
  }
}