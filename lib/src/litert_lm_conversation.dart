import 'dart:async';

import 'package:flutter/services.dart';

import 'models.dart';

/// A stateful conversation session backed by a [LiteRtLmEngine].
///
/// Tracks conversation history across multiple turns. Multiple conversations
/// may share the same underlying engine.
///
/// Obtain an instance via [LiteRtLmEngine.createConversation].
///
/// ## Tool calling
///
/// If the engine was created with tools registered in [ConversationConfig], the
/// model may respond with [LiteRtMessageResponse.hasToolCalls] == `true` instead
/// of producing a text answer. In that case:
///
/// ```dart
/// final res = await conversation.sendText('What is the weather in London?');
/// if (res.hasToolCalls) {
///   final toolResponses = res.toolCalls.map((call) {
///     final args = call.arguments; // Map<String, dynamic>
///     final result = myExecuteTool(call.name, args);
///     return LiteRtContent.toolResponse(call.name, jsonEncode(result));
///   }).toList();
///   final finalResponse = await conversation.sendToolResponses(toolResponses);
///   print(finalResponse.text);
/// }
/// ```
class LiteRtLmConversation {
  final String _conversationId;
  final MethodChannel _channel;
  bool _closed = false;

  // ignore: library_private_types_in_public_api
  LiteRtLmConversation.internal(this._conversationId, this._channel);

  // ──────────────────────────────────────────
  // Send — blocking
  // ──────────────────────────────────────────

  /// Sends a multimodal [message] and waits for the complete response.
  ///
  /// For text-only messages, prefer [sendText] as a convenience shorthand.
  ///
  /// Returns a [LiteRtMessageResponse] with the model's text and/or tool calls.
  /// Throws [LiteRtLmException] on failure.
  Future<LiteRtMessageResponse> sendMessage(List<LiteRtContent> parts) async {
    _assertOpen();
    try {
      final result = await _channel.invokeMethod(
        'conversation/sendMessage',
        {
          'conversationId': _conversationId,
          'parts': parts.map((p) => p.toMap()).toList(),
        },
      ) as Map<Object?, Object?>;
      return LiteRtMessageResponse.fromMap(result);
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'sendMessage failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Convenience method: sends a plain text message and waits for the response.
  Future<LiteRtMessageResponse> sendText(String text) =>
      sendMessage([LiteRtContent.text(text)]);

  // ──────────────────────────────────────────
  // Send — streaming
  // ──────────────────────────────────────────

  /// Sends a multimodal [message] and streams the response token-by-token.
  ///
  /// The returned [Stream] emits text-chunk [LiteRtMessageResponse] objects as
  /// the model generates them. Each intermediate event has an empty [toolCalls]
  /// list. If the model finishes with tool calls instead of — or after — text,
  /// the final event in the stream will have [LiteRtMessageResponse.hasToolCalls]
  /// == `true`.
  ///
  /// Only one streaming request may be active per conversation at a time.
  /// Call [cancel] to abort an ongoing stream.
  ///
  /// ```dart
  /// await for (final event in conversation.sendMessageStream([
  ///   LiteRtContent.imageFile('/path/to/photo.jpg'),
  ///   LiteRtContent.text('What is in this image?'),
  /// ])) {
  ///   if (event.hasToolCalls) {
  ///     // handle tool calls
  ///   } else {
  ///     print(event.text);
  ///   }
  /// }
  /// ```
  Stream<LiteRtMessageResponse> sendMessageStream(List<LiteRtContent> parts) {
    _assertOpen();
    final controller = StreamController<LiteRtMessageResponse>();

    Future<void> startStream() async {
      try {
        // Ask Kotlin to start async generation; it returns the EventChannel name.
        final streamChannelName =
            await _channel.invokeMethod<String>(
          'conversation/sendMessageStream',
          {
            'conversationId': _conversationId,
            'parts': parts.map((p) => p.toMap()).toList(),
          },
        );

        // Subscribe to the per-conversation EventChannel.
        final eventChannel = EventChannel(streamChannelName!);
        final subscription = eventChannel.receiveBroadcastStream().listen(
          (event) {
            if (controller.isClosed) return;
            // Text-chunk events are plain Strings.
            // Tool-call terminal events are Maps with '_type' == 'toolCalls'.
            if (event is String) {
              controller.add(
                LiteRtMessageResponse(text: event, toolCalls: const []),
              );
            } else if (event is Map) {
              final type = event['_type'] as String?;
              if (type == 'toolCalls') {
                final rawCalls = event['calls'] as List<Object?>? ?? const [];
                final toolCalls = rawCalls
                    .cast<Map<Object?, Object?>>()
                    .map(
                      (c) {
                        final args = c['argumentsJson'];
                        return LiteRtToolCall(
                          name: c['name'] as String,
                          argumentsJson: args is String ? args : jsonEncode(args),
                        );
                      },
                    )
                    .toList(growable: false);
                controller.add(
                  LiteRtMessageResponse(text: '', toolCalls: toolCalls),
                );
              }
            }
          },
          onError: (Object error) {
            if (!controller.isClosed) {
              controller.addError(LiteRtLmException(error.toString()));
            }
          },
          onDone: () {
            if (!controller.isClosed) controller.close();
          },
          cancelOnError: true,
        );

        controller.onCancel = () {
          subscription.cancel();
        };
      } on PlatformException catch (e) {
        controller.addError(LiteRtLmException(
          e.message ?? 'sendMessageStream failed',
          code: e.code,
          details: e.details,
        ));
        await controller.close();
      }
    }

    startStream();
    return controller.stream;
  }

  /// Convenience method: streams the response to a plain text message.
  Stream<LiteRtMessageResponse> sendTextStream(String text) =>
      sendMessageStream([LiteRtContent.text(text)]);

  // ──────────────────────────────────────────
  // Tool responses
  // ──────────────────────────────────────────

  /// Sends tool-execution results back to the model and waits for its final
  /// answer.
  ///
  /// Call this after receiving a [LiteRtMessageResponse] with
  /// [LiteRtMessageResponse.hasToolCalls] == `true`. Build each element of
  /// [toolResponseParts] using [LiteRtContent.toolResponse].
  ///
  /// ```dart
  /// final finalResponse = await conversation.sendToolResponses([
  ///   LiteRtContent.toolResponse('getCurrentWeather', jsonEncode({'temp': 22})),
  /// ]);
  /// print(finalResponse.text);
  /// ```
  Future<LiteRtMessageResponse> sendToolResponses(
    List<LiteRtContent> toolResponseParts,
  ) async {
    _assertOpen();
    try {
      final result = await _channel.invokeMethod(
        'conversation/sendMessage',
        {
          'conversationId': _conversationId,
          'role': 'tool',
          'parts': toolResponseParts.map((p) => p.toMap()).toList(),
        },
      ) as Map<Object?, Object?>;
      return LiteRtMessageResponse.fromMap(result);
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'sendToolResponses failed',
        code: e.code,
        details: e.details,
      );
    }
  }
  // ──────────────────────────────────────────
  // Cancel
  // ──────────────────────────────────────────

  /// Cancels any ongoing inference for this conversation.
  ///
  /// Safe to call even if no inference is running.
  Future<void> cancel() async {
    if (_closed) return;
    try {
      await _channel.invokeMethod<void>(
        'conversation/cancel',
        {'conversationId': _conversationId},
      );
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'cancel failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  // ──────────────────────────────────────────
  // Close
  // ──────────────────────────────────────────

  /// Releases native resources for this conversation.
  ///
  /// The conversation cannot be used after calling [close].
  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    try {
      await _channel.invokeMethod<void>(
        'conversation/close',
        {'conversationId': _conversationId},
      );
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'conversation close failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  void _assertOpen() {
    if (_closed) {
      throw StateError(
        'Conversation is closed. Create a new conversation via engine.createConversation().',
      );
    }
  }
}
