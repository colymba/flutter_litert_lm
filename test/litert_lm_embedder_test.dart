import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ai_edge_litert_lm/ai_edge_litert_lm.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const channel = MethodChannel('flutter_litert_lm');

  /// Installs a mock platform handler and records incoming calls.
  List<MethodCall> mockPlatform({
    Map<String, dynamic> Function()? onInitialize,
    Float32List Function(MethodCall call)? onEmbed,
    List<Float32List> Function(MethodCall call)? onEmbedBatch,
  }) {
    final log = <MethodCall>[];
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      log.add(call);
      switch (call.method) {
        case 'embedder/initialize':
          return (onInitialize?.call() ??
              {
                'embedderId': 'embedder_1',
                'dimension': 768,
                'maxSequenceLength': 256,
              });
        case 'embedder/embed':
          return onEmbed?.call(call) ?? Float32List.fromList([0.6, 0.8]);
        case 'embedder/embedBatch':
          return onEmbedBatch?.call(call) ??
              [
                Float32List.fromList([1.0, 0.0]),
                Float32List.fromList([0.0, 1.0]),
              ];
        case 'embedder/close':
          return null;
      }
      throw MissingPluginException(call.method);
    });
    return log;
  }

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  // ──────────────────────────────────────────────────────────────────────
  // EmbedderConfig
  // ──────────────────────────────────────────────────────────────────────

  group('EmbedderConfig', () {
    test('toMap contains required fields and defaults', () {
      const config = EmbedderConfig(
        modelPath: '/models/embeddinggemma.tflite',
        tokenizerPath: '/models/sentencepiece.model',
      );
      final map = config.toMap();

      expect(map['modelPath'], '/models/embeddinggemma.tflite');
      expect(map['tokenizerPath'], '/models/sentencepiece.model');
      expect(map['backend'], 'cpu'); // default
      expect(map['normalize'], isTrue); // default
    });

    test('toMap reflects overrides', () {
      const config = EmbedderConfig(
        modelPath: '/m.tflite',
        tokenizerPath: '/t.model',
        backend: BackendType.gpu,
        normalize: false,
      );
      final map = config.toMap();

      expect(map['backend'], 'gpu');
      expect(map['normalize'], isFalse);
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtLmEmbedder
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtLmEmbedder', () {
    const config = EmbedderConfig(
      modelPath: '/m.tflite',
      tokenizerPath: '/t.model',
    );

    test('initialize stores id, dimension and maxSequenceLength', () async {
      mockPlatform();
      final embedder = LiteRtLmEmbedder();

      expect(embedder.isInitialized, isFalse);
      await embedder.initialize(config);

      expect(embedder.isInitialized, isTrue);
      expect(embedder.dimension, 768);
      expect(embedder.maxSequenceLength, 256);
    });

    test('initialize twice throws StateError', () async {
      mockPlatform();
      final embedder = LiteRtLmEmbedder();
      await embedder.initialize(config);

      expect(() => embedder.initialize(config), throwsStateError);
    });

    test('embed before initialize throws StateError', () {
      mockPlatform();
      final embedder = LiteRtLmEmbedder();

      expect(() => embedder.embed('hello'), throwsStateError);
      expect(() => embedder.dimension, throwsStateError);
    });

    test('embed sends text and promptPrefix, returns Float32List', () async {
      final log = mockPlatform();
      final embedder = LiteRtLmEmbedder();
      await embedder.initialize(config);

      final vector = await embedder.embed(
        'hello world',
        promptPrefix: LiteRtLmEmbedder.queryPrefix,
      );

      expect(vector, isA<Float32List>());
      expect(vector, Float32List.fromList([0.6, 0.8]));

      final call = log.singleWhere((c) => c.method == 'embedder/embed');
      final args = (call.arguments as Map).cast<String, dynamic>();
      expect(args['embedderId'], 'embedder_1');
      expect(args['text'], 'hello world');
      expect(args['promptPrefix'], LiteRtLmEmbedder.queryPrefix);
    });

    test('embed omits promptPrefix when not given', () async {
      final log = mockPlatform();
      final embedder = LiteRtLmEmbedder();
      await embedder.initialize(config);

      await embedder.embed('hello');

      final call = log.singleWhere((c) => c.method == 'embedder/embed');
      final args = (call.arguments as Map).cast<String, dynamic>();
      expect(args.containsKey('promptPrefix'), isFalse);
    });

    test('embedBatch returns one vector per input, in order', () async {
      final log = mockPlatform();
      final embedder = LiteRtLmEmbedder();
      await embedder.initialize(config);

      final vectors = await embedder.embedBatch(['a', 'b']);

      expect(vectors, hasLength(2));
      expect(vectors[0], [1.0, 0.0]);
      expect(vectors[1], [0.0, 1.0]);

      final call = log.singleWhere((c) => c.method == 'embedder/embedBatch');
      final args = (call.arguments as Map).cast<String, dynamic>();
      expect(args['texts'], ['a', 'b']);
    });

    test('close resets state and allows re-initialization', () async {
      mockPlatform();
      final embedder = LiteRtLmEmbedder();
      await embedder.initialize(config);

      await embedder.close();
      expect(embedder.isInitialized, isFalse);

      await embedder.initialize(config);
      expect(embedder.isInitialized, isTrue);
    });

    test('close when not initialized is a no-op', () async {
      final log = mockPlatform();
      final embedder = LiteRtLmEmbedder();

      await embedder.close();
      expect(log, isEmpty);
    });

    test('PlatformException is wrapped in LiteRtLmException', () async {
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
          .setMockMethodCallHandler(channel, (call) async {
        throw PlatformException(code: 'EMBEDDER_INIT_FAILED', message: 'boom');
      });
      final embedder = LiteRtLmEmbedder();

      expect(
        () => embedder.initialize(config),
        throwsA(
          isA<LiteRtLmException>()
              .having((e) => e.code, 'code', 'EMBEDDER_INIT_FAILED')
              .having((e) => e.message, 'message', 'boom'),
        ),
      );
    });

    test('EmbeddingGemma prompt prefixes match the model card', () {
      expect(LiteRtLmEmbedder.documentPrefix, 'title: none | text: ');
      expect(LiteRtLmEmbedder.queryPrefix, 'task: search result | query: ');
    });
  });
}
