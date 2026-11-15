import 'dart:convert';
import 'dart:math' as math;
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:hand_landmarker/hand_landmarker.dart';
import 'classifier.dart';
import 'features.dart';

// Set from the training-data check: -1 mirrors hands whose cross product is positive.
const double kTrainSign = -1.0;

// Dot placement. If the dots land in the wrong place, flip these (see notes).
const bool kSwapAxes = true; // screen-x comes from landmark y, screen-y from landmark x
const bool kInvertX = true;
const bool kInvertY = true;

// Which landmarks to highlight for each part of a feature name (as in the desktop demo).
const Map<String, List<int>> kJoints = {
  'index': [5, 6, 7, 8], 'idx': [5, 6, 7, 8], 'idxbase': [5, 6, 7, 8],
  'middle': [9, 10, 11, 12], 'mid': [9, 10, 11, 12], 'midbase': [9, 10, 11, 12],
  'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20],
  'thumb': [1, 2, 3, 4], 'th': [1, 2, 3, 4],
};

void main() => runApp(const MaterialApp(home: MudraScreen()));

class Analysis {
  final String text;
  final Set<int> hl;
  Analysis(this.text, this.hl);
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
  MudraClassifier? _clf;
  Map<String, dynamic>? _stats;
  String _status = 'starting...';
  String _target = 'Pathaka';

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
      if (mounted) setState(() => _status = 'ready');
    } catch (e) {
      if (mounted) setState(() => _status = 'error: $e');
    }
  }

  @override
  void dispose() {
    _controller?.stopImageStream();
    _controller?.dispose();
    _plugin?.dispose();
    super.dispose();
  }

  Analysis _analyze(List<Hand> hands, Size ps) {
    if (hands.isEmpty) return Analysis('no hand', {});
    final clf = _clf!;
    // Landmarks are normalized in the sensor frame; training used x*w, y*h, z*w.
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
    return Analysis(
        'pred ${clf.classes[best]} ${pr[best].toStringAsFixed(2)}\n'
        '$_target P=${pt.toStringAsFixed(2)}\n'
        'mirrored=$mirrored\n'
        '${fix.$1}',
        fix.$2);
  }

  (String, Set<int>) _fix(List<List<double>> p) {
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
    if (top == null) return ('', <int>{});
    if (topZ.abs() < 3.0) return ('looks good (${topZ.abs().toStringAsFixed(1)})', <int>{});
    final high = topZ > 0;
    final hl = <int>{};
    for (final t in top.split('_')) {
      hl.addAll(kJoints[t] ?? const <int>[]);
    }
    return (
      'fix: $top ${high ? 'high' : 'low'} (${topZ.toStringAsFixed(1)})\n'
          '${_hint(top, high)}',
      hl
    );
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
    final up = Size(ps.height, ps.width); // upright (portrait) preview size
    return Scaffold(
      body: LayoutBuilder(builder: (context, box) {
        final sw = box.maxWidth, sh = box.maxHeight;
        final scale = math.max(sw / up.width, sh / up.height);
        final ox = (sw - up.width * scale) / 2;
        final oy = (sh - up.height * scale) / 2;
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
          Positioned.fill(
            child: StreamBuilder<List<Hand>>(
              stream: _plugin!.landmarkStream,
              initialData: const [],
              builder: (context, snap) {
                final hands = snap.data ?? [];
                final a = _analyze(hands, ps);
                final pts = <Offset>[];
                if (hands.isNotEmpty) {
                  for (final l in hands.first.landmarks) {
                    var fx = kSwapAxes ? l.y : l.x;
                    var fy = kSwapAxes ? l.x : l.y;
                    if (kInvertX) fx = 1 - fx;
                    if (kInvertY) fy = 1 - fy;
                    pts.add(Offset(ox + fx * up.width * scale,
                        oy + fy * up.height * scale));
                  }
                }
                return Stack(children: [
                  Positioned.fill(
                      child: CustomPaint(painter: DotPainter(pts, a.hl))),
                  Positioned(
                    left: 8,
                    right: 8,
                    top: 40,
                    child: Container(
                        color: Colors.black54,
                        padding: const EdgeInsets.all(8),
                        child: Text(a.text,
                            style: const TextStyle(
                                color: Colors.white, fontSize: 16))),
                  ),
                ]);
              },
            ),
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
                onChanged: (v) => setState(() => _target = v!),
              ),
            ),
          ),
        ]);
      }),
    );
  }
}