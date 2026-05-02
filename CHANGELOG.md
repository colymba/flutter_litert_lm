## 0.2.1

* Re-verified native dependencies. Android uses Gradle latest.release for litertlm-android.

## 0.2.0

* **Tool calling support** — register Dart-defined tools on a conversation and let the
  model invoke them during inference.
* New `ConversationConfig.tools` field accepts a list of `LiteRtToolDeclaration` objects.
  When tools are provided, `automaticToolCalling` is disabled automatically so every tool
  call is routed back to Dart.
* New `LiteRtToolDeclaration` — describe a function with an OpenAPI-style JSON Schema.
* New `LiteRtToolCall` — carries the tool name and JSON-encoded arguments from the model.
* New `LiteRtMessageResponse` — unified return type for all `sendMessage`/`sendText` calls,
  exposing both `.text` and `.toolCalls`. Replaces the previous bare `String` return.
* New `LiteRtContent.toolResponse(name, json)` factory — feed tool results back to the model.
* New `LiteRtLmConversation.sendToolResponses(parts)` — sends tool results and returns the
  model's final answer.
* Streaming: terminal tool-call events are emitted on the `EventChannel` stream before close.
* Android: added `DartOpenApiTool` Kotlin bridge; `ConversationHandler` updated to wire tools
  into `ConversationConfig` and serialise/deserialise `Message.toolCalls`.
* `flutter analyze` clean; 24 unit tests (7 new groups covering all tool-calling types).

## 0.1.0

* Initial Android implementation via the official `litertlm-android` Kotlin SDK.
* Engine lifecycle: `initialize`, `close`.
* Conversation lifecycle: `create`, `sendMessage` (blocking), `sendMessageStream` (streaming), `cancel`, `close`.
* Full multimodal input support: text, image file path, image bytes, audio file path, audio bytes.
* Configurable backends: CPU, GPU, NPU.
* iOS stub (throws `UnsupportedError` on all calls; full support pending Swift SDK availability).
