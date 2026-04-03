## 0.1.0

* Initial Android implementation via the official `litertlm-android` Kotlin SDK.
* Engine lifecycle: `initialize`, `close`.
* Conversation lifecycle: `create`, `sendMessage` (blocking), `sendMessageStream` (streaming), `cancel`, `close`.
* Full multimodal input support: text, image file path, image bytes, audio file path, audio bytes.
* Configurable backends: CPU, GPU, NPU.
* iOS stub (throws `UnsupportedError` on all calls; full support pending Swift SDK availability).
