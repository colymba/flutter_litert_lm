import 'dart:typed_data';

// ─────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────

/// The hardware backend to use for inference.
enum BackendType {
  /// CPU backend (always available).
  cpu,

  /// GPU backend (recommended for best performance when available).
  gpu,

  /// NPU / hardware accelerator backend.
  npu;

  String get _value => name;
}

// ─────────────────────────────────────────────
// EngineConfig
// ─────────────────────────────────────────────

/// Configuration for initialising a [LiteRtLmEngine].
class EngineConfig {
  /// Path to the `.litertlm` model file on the device.
  final String modelPath;

  /// Primary inference backend. Defaults to [BackendType.gpu].
  final BackendType backend;

  /// Optional vision/image backend. Required for multimodal image inference.
  final BackendType? visionBackend;

  /// Optional audio backend. Required for multimodal audio inference.
  final BackendType? audioBackend;

  /// Optional directory for caching compiled model artifacts to speed up
  /// subsequent loads.
  final String? cacheDir;

  /// Maximum number of tokens (context window). Uses the model default when
  /// null.
  final int? maxNumTokens;

  const EngineConfig({
    required this.modelPath,
    this.backend = BackendType.gpu,
    this.visionBackend,
    this.audioBackend,
    this.cacheDir,
    this.maxNumTokens,
  });

  Map<String, dynamic> toMap() => {
        'modelPath': modelPath,
        'backend': backend._value,
        if (visionBackend != null) 'visionBackend': visionBackend!._value,
        if (audioBackend != null) 'audioBackend': audioBackend!._value,
        if (cacheDir != null) 'cacheDir': cacheDir,
        if (maxNumTokens != null) 'maxNumTokens': maxNumTokens,
      };
}

// ─────────────────────────────────────────────
// ConversationConfig
// ─────────────────────────────────────────────

/// Configuration for creating a [LiteRtLmConversation].
class ConversationConfig {
  /// Optional system-level instruction prepended to every conversation.
  final String? systemInstruction;

  /// Top-K sampling parameter. Uses model default when null.
  final int? topK;

  /// Top-P (nucleus) sampling parameter. Uses model default when null.
  final double? topP;

  /// Sampling temperature. Uses model default when null.
  final double? temperature;

  /// Maximum number of tokens to generate per response.
  final int? maxOutputTokens;

  const ConversationConfig({
    this.systemInstruction,
    this.topK,
    this.topP,
    this.temperature,
    this.maxOutputTokens,
  });

  Map<String, dynamic> toMap() => {
        if (systemInstruction != null) 'systemInstruction': systemInstruction,
        if (topK != null) 'topK': topK,
        if (topP != null) 'topP': topP,
        if (temperature != null) 'temperature': temperature,
        if (maxOutputTokens != null) 'maxOutputTokens': maxOutputTokens,
      };
}

// ─────────────────────────────────────────────
// LiteRtContent — sealed multimodal content
// ─────────────────────────────────────────────

/// A single piece of multimodal content to send in a message.
///
/// Use the factory constructors:
/// - [LiteRtContent.text]
/// - [LiteRtContent.imageFile]
/// - [LiteRtContent.imageBytes]
/// - [LiteRtContent.audioFile]
/// - [LiteRtContent.audioBytes]
sealed class LiteRtContent {
  const LiteRtContent();

  /// A plain text part.
  const factory LiteRtContent.text(String text) = LiteRtTextContent;

  /// An image given by its absolute file path on the device.
  const factory LiteRtContent.imageFile(String path) = LiteRtImageFileContent;

  /// An image given as raw bytes (e.g. from `dart:io` or `image_picker`).
  factory LiteRtContent.imageBytes(Uint8List bytes) =
      LiteRtImageBytesContent;

  /// An audio clip given by its absolute file path on the device.
  const factory LiteRtContent.audioFile(String path) = LiteRtAudioFileContent;

  /// An audio clip given as raw bytes.
  factory LiteRtContent.audioBytes(Uint8List bytes) =
      LiteRtAudioBytesContent;

  /// Serialises this content part to a map for the MethodChannel codec.
  Map<String, dynamic> toMap();
}

/// Plain text content.
final class LiteRtTextContent extends LiteRtContent {
  final String text;
  const LiteRtTextContent(this.text);

  @override
  Map<String, dynamic> toMap() => {'type': 'text', 'text': text};
}

/// Image content from a file path.
final class LiteRtImageFileContent extends LiteRtContent {
  final String path;
  const LiteRtImageFileContent(this.path);

  @override
  Map<String, dynamic> toMap() => {'type': 'imageFile', 'path': path};
}

/// Image content from raw bytes.
final class LiteRtImageBytesContent extends LiteRtContent {
  final Uint8List bytes;
  LiteRtImageBytesContent(this.bytes);

  @override
  Map<String, dynamic> toMap() => {'type': 'imageBytes', 'data': bytes};
}

/// Audio content from a file path.
final class LiteRtAudioFileContent extends LiteRtContent {
  final String path;
  const LiteRtAudioFileContent(this.path);

  @override
  Map<String, dynamic> toMap() => {'type': 'audioFile', 'path': path};
}

/// Audio content from raw bytes.
final class LiteRtAudioBytesContent extends LiteRtContent {
  final Uint8List bytes;
  LiteRtAudioBytesContent(this.bytes);

  @override
  Map<String, dynamic> toMap() => {'type': 'audioBytes', 'data': bytes};
}

// ─────────────────────────────────────────────
// LiteRtMessage
// ─────────────────────────────────────────────

/// A message in a conversation turn, consisting of one or more content parts.
class LiteRtMessage {
  /// The role of the message author: `'user'`, `'model'`, or `'system'`.
  final String role;

  /// The content parts — at minimum one [LiteRtContent.text], but can include
  /// image or audio content for multimodal models.
  final List<LiteRtContent> parts;

  const LiteRtMessage({
    required this.role,
    required this.parts,
  });

  /// Convenience constructor for a simple text-only user message.
  factory LiteRtMessage.user(String text) => LiteRtMessage(
        role: 'user',
        parts: [LiteRtContent.text(text)],
      );

  /// Convenience constructor for a simple text-only model message.
  factory LiteRtMessage.model(String text) => LiteRtMessage(
        role: 'model',
        parts: [LiteRtContent.text(text)],
      );

  Map<String, dynamic> toMap() => {
        'role': role,
        'parts': parts.map((p) => p.toMap()).toList(),
      };
}

// ─────────────────────────────────────────────
// Exceptions
// ─────────────────────────────────────────────

/// Thrown when a native LiteRT-LM operation fails.
class LiteRtLmException implements Exception {
  final String message;
  final String? code;
  final dynamic details;

  const LiteRtLmException(this.message, {this.code, this.details});

  @override
  String toString() =>
      'LiteRtLmException(${code != null ? '$code: ' : ''}$message)';
}
