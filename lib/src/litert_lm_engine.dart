import 'package:flutter/services.dart';

import 'litert_lm_conversation.dart';
import 'models.dart';

/// Entry point for LiteRT-LM inference.
///
/// Create one [LiteRtLmEngine] per model. The engine is a heavyweight object
/// that holds model weights — create it once and reuse it across multiple
/// conversations.
///
/// ```dart
/// final engine = LiteRtLmEngine();
/// await engine.initialize(EngineConfig(
///   modelPath: '/path/to/model.litertlm',
///   backend: BackendType.gpu,
/// ));
///
/// final conversation = await engine.createConversation();
/// // ... use the conversation ...
///
/// await conversation.close();
/// await engine.close();
/// ```
class LiteRtLmEngine {
  static const _channel = MethodChannel('flutter_litert_lm');

  String? _engineId;

  /// Whether this engine has been successfully initialised.
  bool get isInitialized => _engineId != null;

  /// Initialises the engine with the given [config].
  ///
  /// This is a blocking call that loads the model into memory — call it on a
  /// background isolate or ensure the UI is showing a loading state.
  ///
  /// Throws [LiteRtLmException] if initialisation fails.
  /// Throws [StateError] if already initialised.
  Future<void> initialize(EngineConfig config) async {
    if (_engineId != null) {
      throw StateError('Engine is already initialised. Call close() first.');
    }
    try {
      final result = await _channel.invokeMethod<String>(
        'engine/initialize',
        config.toMap(),
      );
      _engineId = result;
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Engine initialisation failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Creates a new stateful [LiteRtLmConversation] from this engine.
  ///
  /// Each conversation tracks its own history and can be used independently.
  /// Multiple conversations may be created from the same engine.
  ///
  /// Throws [LiteRtLmException] if creation fails.
  /// Throws [StateError] if engine is not initialised.
  Future<LiteRtLmConversation> createConversation([
    ConversationConfig? config,
  ]) async {
    _assertInitialized();
    try {
      final args = <String, dynamic>{'engineId': _engineId!};
      if (config != null) args.addAll(config.toMap());

      final conversationId = await _channel.invokeMethod<String>(
        'conversation/create',
        args,
      );
      return LiteRtLmConversation.internal(conversationId!, _channel);
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Conversation creation failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Releases all native resources held by this engine.
  ///
  /// All conversations created from this engine must be closed first.
  Future<void> close() async {
    if (_engineId == null) return;
    try {
      await _channel.invokeMethod<void>(
        'engine/close',
        {'engineId': _engineId!},
      );
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'Engine close failed',
        code: e.code,
        details: e.details,
      );
    } finally {
      _engineId = null;
    }
  }

  void _assertInitialized() {
    if (_engineId == null) {
      throw StateError(
        'Engine is not initialised. Call initialize() first.',
      );
    }
  }
}
