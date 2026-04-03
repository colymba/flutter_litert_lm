import 'dart:async';

import 'package:flutter/services.dart';

import 'models.dart';

/// A stateful conversation session backed by a [LiteRtLmEngine].
///
/// Tracks conversation history across multiple turns. Multiple conversations
/// may share the same underlying engine.
///
/// Obtain an instance via [LiteRtLmEngine.createConversation].
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
  /// Returns the model's text response.
  /// Throws [LiteRtLmException] on failure.
  Future<String> sendMessage(List<LiteRtContent> parts) async {
    _assertOpen();
    try {
      final result = await _channel.invokeMethod<String>(
        'conversation/sendMessage',
        {
          'conversationId': _conversationId,
          'parts': parts.map((p) => p.toMap()).toList(),
        },
      );
      return result ?? '';
    } on PlatformException catch (e) {
      throw LiteRtLmException(
        e.message ?? 'sendMessage failed',
        code: e.code,
        details: e.details,
      );
    }
  }

  /// Convenience method: sends a plain text message and waits for the response.
  Future<String> sendText(String text) =>
      sendMessage([LiteRtContent.text(text)]);

  // ──────────────────────────────────────────
  // Send — streaming
  // ──────────────────────────────────────────

  /// Sends a multimodal [message] and streams the response token-by-token.
  ///
  /// The returned [Stream] emits text chunks as they are generated.  The
  /// stream closes normally when generation completes and emits an error if
  /// the native layer reports one.
  ///
  /// Only one streaming request may be active per conversation at a time.
  /// Call [cancel] to abort an ongoing stream.
  ///
  /// ```dart
  /// await for (final chunk in conversation.sendMessageStream([
  ///   LiteRtContent.imageFile('/path/to/photo.jpg'),
  ///   LiteRtContent.text('What is in this image?'),
  /// ])) {
  ///   print(chunk);
  /// }
  /// ```
  Stream<String> sendMessageStream(List<LiteRtContent> parts) {
    _assertOpen();
    final controller = StreamController<String>();

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
            if (!controller.isClosed) {
              controller.add(event as String);
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
  Stream<String> sendTextStream(String text) =>
      sendMessageStream([LiteRtContent.text(text)]);

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
