import 'dart:convert';
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

  /// Tools that the model may invoke during this conversation.
  ///
  /// When non-empty, the native layer automatically sets
  /// `automaticToolCalling = false` so that every tool call is routed back to
  /// Dart for manual execution via [LiteRtLmConversation.sendToolResponses].
  final List<LiteRtToolDeclaration> tools;

  const ConversationConfig({
    this.systemInstruction,
    this.topK,
    this.topP,
    this.temperature,
    this.maxOutputTokens,
    this.tools = const [],
  });

  Map<String, dynamic> toMap() => {
        if (systemInstruction != null) 'systemInstruction': systemInstruction,
        if (topK != null) 'topK': topK,
        if (topP != null) 'topP': topP,
        if (temperature != null) 'temperature': temperature,
        if (maxOutputTokens != null) 'maxOutputTokens': maxOutputTokens,
        // Always disable automatic execution — tools are handled in Dart.
        if (tools.isNotEmpty) 'tools': tools.map((t) => t.toMap()).toList(),
        if (tools.isNotEmpty) 'automaticToolCalling': false,
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
  factory LiteRtContent.imageBytes(Uint8List bytes) = LiteRtImageBytesContent;

  /// An audio clip given by its absolute file path on the device.
  const factory LiteRtContent.audioFile(String path) = LiteRtAudioFileContent;

  /// An audio clip given as raw bytes.
  factory LiteRtContent.audioBytes(Uint8List bytes) = LiteRtAudioBytesContent;

  /// A tool-execution result to send back to the model.
  ///
  /// [toolName] must match the name of the [LiteRtToolDeclaration] that was
  /// called. [responseJson] is a JSON string containing the result.
  const factory LiteRtContent.toolResponse(
    String toolName,
    String responseJson,
  ) = LiteRtToolResponseContent;

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

/// A tool-execution result part sent back to the model.
final class LiteRtToolResponseContent extends LiteRtContent {
  final String toolName;
  final String responseJson;
  const LiteRtToolResponseContent(this.toolName, this.responseJson);

  @override
  Map<String, dynamic> toMap() => {
        'type': 'toolResponse',
        'toolName': toolName,
        'responseJson': responseJson,
      };
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
// LiteRtToolDeclaration
// ─────────────────────────────────────────────

/// Declares a tool (function) that the model may call during a conversation.
///
/// Tools are registered on [ConversationConfig.tools]. The model will decide
/// when to call them based on the conversation context.
///
/// [parameters] must be a JSON Schema *object* describing the function's
/// arguments — the same schema format used by the OpenAPI / Gemini function
/// calling spec. Example:
///
/// ```dart
/// LiteRtToolDeclaration(
///   name: 'getCurrentWeather',
///   description: 'Returns the current weather for a city.',
///   parameters: {
///     'type': 'object',
///     'properties': {
///       'city': {'type': 'string', 'description': 'City name, e.g. London'},
///       'unit': {
///         'type': 'string',
///         'enum': ['celsius', 'fahrenheit'],
///         'description': 'Temperature unit. Default: celsius',
///       },
///     },
///     'required': ['city'],
///   },
/// )
/// ```
class LiteRtToolDeclaration {
  /// The function name the model will use to invoke this tool.
  final String name;

  /// Human-readable description used by the model to decide when to call this
  /// tool. Be concise and precise.
  final String description;

  /// JSON Schema *object* describing the function parameters.
  final Map<String, dynamic> parameters;

  const LiteRtToolDeclaration({
    required this.name,
    required this.description,
    required this.parameters,
  });

  /// Serialises the full OpenAPI function spec to a JSON string.
  String toOpenApiJson() => jsonEncode({
        'name': name,
        'description': description,
        'parameters': parameters,
      });

  Map<String, dynamic> toMap() => {
        'name': name,
        'description': description,
        'parametersJson': toOpenApiJson(),
      };
}

// ─────────────────────────────────────────────
// LiteRtToolCall
// ─────────────────────────────────────────────

/// A tool invocation requested by the model.
///
/// Returned as part of [LiteRtMessageResponse.toolCalls] when the model
/// determines it needs to call one or more tools before answering.
///
/// After executing the tool, feed the result back via
/// [LiteRtLmConversation.sendToolResponses].
class LiteRtToolCall {
  /// The name of the tool to call — matches a [LiteRtToolDeclaration.name].
  final String name;

  /// Raw JSON string of the argument object the model wants to pass.
  /// Parse with `jsonDecode(argumentsJson)` to get a `Map<String, dynamic>`.
  final String argumentsJson;

  const LiteRtToolCall({required this.name, required this.argumentsJson});

  /// Convenience: decode [argumentsJson] into a typed map.
  Map<String, dynamic> get arguments =>
      (jsonDecode(argumentsJson) as Map).cast<String, dynamic>();

  @override
  String toString() => 'LiteRtToolCall(name: $name, arguments: $argumentsJson)';
}

// ─────────────────────────────────────────────
// LiteRtMessageResponse
// ─────────────────────────────────────────────

/// The result of a [LiteRtLmConversation.sendMessage] or
/// [LiteRtLmConversation.sendText] call.
///
/// In the common text-only case [toolCalls] is empty and [text] holds the
/// model's answer. When the model needs to call tools, [toolCalls] is
/// populated and [text] may be empty — you should execute each tool and call
/// [LiteRtLmConversation.sendToolResponses] to get the final answer.
class LiteRtMessageResponse {
  /// The model's text output. May be empty when [hasToolCalls] is `true`.
  final String text;

  /// Tool invocations requested by the model. Empty for text-only responses.
  final List<LiteRtToolCall> toolCalls;

  const LiteRtMessageResponse({
    required this.text,
    required this.toolCalls,
  });

  /// `true` when the model is requesting one or more tool executions.
  bool get hasToolCalls => toolCalls.isNotEmpty;

  /// Decodes from the map returned by the native MethodChannel.
  factory LiteRtMessageResponse.fromMap(Map<Object?, Object?> map) {
    final text = (map['text'] as String?) ?? '';
    final rawCalls = map['toolCalls'] as List<Object?>? ?? const [];
    final toolCalls = rawCalls
        .cast<Map<Object?, Object?>>()
        .map(
          (c) => LiteRtToolCall(
            name: c['name'] as String,
            argumentsJson: c['argumentsJson'] as String,
          ),
        )
        .toList(growable: false);
    return LiteRtMessageResponse(text: text, toolCalls: toolCalls);
  }

  @override
  String toString() =>
      'LiteRtMessageResponse(text: $text, toolCalls: $toolCalls)';
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
