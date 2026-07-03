import 'dart:typed_data';

import 'package:flutter/services.dart';

import 'models.dart';

// ─────────────────────────────────────────────
// EmbedderConfig
// ─────────────────────────────────────────────

/// Configuration for initialising a [LiteRtLmEmbedder].
class EmbedderConfig {
  /// Path to the embedding model `.tflite` file on the device
  /// (e.g. an EmbeddingGemma variant from `litert-community`).
  final String modelPath;

  /// Path to the sentencepiece tokenizer model file (`.model` / `.spm`)
  /// matching the embedding model's vocabulary.
  final String tokenizerPath;

  /// Inference backend. Defaults to [BackendType.cpu] — embedding encoders
  /// are small and CPU inference avoids competing with a generation engine
  /// for the GPU.
  final BackendType backend;

  /// Whether to L2-normalize the output vector, so that dot product equals
  /// cosine similarity. Defaults to `true`. (Normalizing an already
  /// normalized vector is a no-op, so this is safe to leave on.)
  final bool normalize;

  const EmbedderConfig({
    required this.modelPath,
    required this.tokenizerPath,
    this.backend = BackendType.cpu,
    this.normalize = true,
  });

  Map<String, dynamic> toMap() => {
        'modelPath': modelPath,
        'tokenizerPath': tokenizerPath,
        'backend': backend.name,
        'normalize': normalize,
      };
}

// ─────────────────────────────────────────────
// LiteRtLmEmbedder
// ─────────────────────────────────────────────

/// On-device text embeddings via the LiteRT interpreter.
///
/// Runs an embedding encoder model (such as EmbeddingGemma) directly on the
/// LiteRT (TFLite) interpreter — independent from [LiteRtLmEngine], which
/// wraps the LiteRT-LM *generation* runtime. The embedder holds its own,
/// much smaller model and can be kept alive without loading a generation
/// model into memory.
///
/// ```dart
/// final embedder = LiteRtLmEmbedder();
/// await embedder.initialize(EmbedderConfig(
///   modelPath: '/path/to/embeddinggemma-300m_seq256.tflite',
///   tokenizerPath: '/path/to/sentencepiece.model',
/// ));
///
/// final docVector = await embedder.embed(
///   'Slept badly, anxious all day.',
///   promptPrefix: LiteRtLmEmbedder.documentPrefix,
/// );
/// final queryVector = await embedder.embed(
///   'when was I anxious?',
///   promptPrefix: LiteRtLmEmbedder.queryPrefix,
/// );
///
/// await embedder.close();
/// ```
///
/// Currently implemented on Android. iOS calls throw a [LiteRtLmException]
/// with code `NOT_SUPPORTED`.
class LiteRtLmEmbedder {
  static const _channel = MethodChannel('flutter_litert_lm');

  /// EmbeddingGemma task prompt for indexing documents.
  ///
  /// EmbeddingGemma was trained with task-specific prompt prefixes; using
  /// them measurably improves retrieval quality. See the model card.
  static const String documentPrefix = 'title: none | text: ';

  /// EmbeddingGemma task prompt for retrieval queries.
  static const String queryPrefix = 'task: search result | query: ';

  String? _embedderId;
  int? _dimension;
  int? _maxSequenceLength;

  /// Whether this embedder has been successfully initialised.
  bool get isInitialized => _embedderId != null;

  /// The dimensionality of the embedding vectors (e.g. 768 for
  /// EmbeddingGemma). Available after [initialize].
  int get dimension {
    _assertInitialized();
    return _dimension!;
  }

  /// The model's fixed input sequence length in tokens (e.g. 256). Longer
  /// inputs are truncated by the native side — chunk long texts in the
  /// caller. Available after [initialize].
  int get maxSequenceLength {
    _assertInitialized();
    return _maxSequenceLength!;
  }

  /// Initialises the embedder with the given [config].
  ///
  /// Loads the `.tflite` model and the sentencepiece tokenizer into memory.
  ///
  /// Throws [LiteRtLmException] if initialisation fails.
  /// Throws [StateError] if already initialised.
  Future<void> initialize(EmbedderConfig config) async {
    if (_embedderId != null) {
      throw StateError('Embedder is already initialised. Call close() first.');
    }
    try {
      final result = await _channel.invokeMapMethod<String, dynamic>(
        'embedder/initialize',
        config.toMap(),
      );
      _embedderId = result!['embedderId'] as String;
      _dimension = result['dimension'] as int;
      _maxSequenceLength = result['maxSequenceLength'] as int;
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Embedder initialisation failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Computes the embedding vector for [text].
  ///
  /// [promptPrefix] is prepended to the text before tokenization. For
  /// EmbeddingGemma pass [documentPrefix] when indexing and [queryPrefix]
  /// when searching (or a custom task prompt).
  ///
  /// Returns a [Float32List] of length [dimension]. Input longer than
  /// [maxSequenceLength] tokens is truncated.
  ///
  /// Throws [LiteRtLmException] if inference fails.
  /// Throws [StateError] if the embedder is not initialised.
  Future<Float32List> embed(String text, {String? promptPrefix}) async {
    _assertInitialized();
    try {
      final result = await _channel.invokeMethod<Float32List>(
        'embedder/embed',
        {
          'embedderId': _embedderId!,
          'text': text,
          if (promptPrefix != null) 'promptPrefix': promptPrefix,
        },
      );
      return result!;
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Embedding failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Computes embeddings for several [texts] in one native call.
  ///
  /// More efficient than repeated [embed] calls for batch indexing (e.g.
  /// the chunks of one long document). Returns one vector per input, in
  /// order.
  Future<List<Float32List>> embedBatch(
    List<String> texts, {
    String? promptPrefix,
  }) async {
    _assertInitialized();
    try {
      final result = await _channel.invokeListMethod<Float32List>(
        'embedder/embedBatch',
        {
          'embedderId': _embedderId!,
          'texts': texts,
          if (promptPrefix != null) 'promptPrefix': promptPrefix,
        },
      );
      return result!;
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Batch embedding failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Releases the native interpreter and tokenizer.
  Future<void> close() async {
    if (_embedderId == null) return;
    try {
      await _channel.invokeMethod<void>(
        'embedder/close',
        {'embedderId': _embedderId!},
      );
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Embedder close failed',
        code: e.code,
        details: e.details,
      );
    } finally {
      _embedderId = null;
      _dimension = null;
      _maxSequenceLength = null;
    }
  }

  void _assertInitialized() {
    if (_embedderId == null) {
      throw StateError(
        'Embedder is not initialised. Call initialize() first.',
      );
    }
  }
}
