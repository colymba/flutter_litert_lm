# flutter_litert_lm

A Flutter plugin for on-device LLM inference using Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) framework.

| Platform | Status |
|---|---|
| Android | ✅ Supported via official `litertlm-android` Kotlin SDK |
| iOS | 🚧 Stub — pending official Swift SDK |

---

## Features

- **Stateful multi-turn conversations** — tracks history automatically.
- **Streaming responses** — receive tokens as they are generated via a `Stream<String>`.
- **Multimodal inputs** — text, images (file path or raw bytes), and audio (file path or raw bytes) in the same message.
- **Configurable backends** — CPU, GPU, or NPU.
- **Multiple concurrent engines and conversations** — manage model lifecycle explicitly.

---

## Android setup

In your app's `android/app/build.gradle`, ensure:

```gradle
android {
    defaultConfig {
        minSdk 26  // required by LiteRT-LM
    }
}
```

The `litertlm-android` dependency is pulled automatically from Google Maven via this plugin's `build.gradle`.

---

## Usage

### 1. Initialize the engine

```dart
import 'package:flutter_litert_lm/flutter_litert_lm.dart';

final engine = LiteRtLmEngine();

await engine.initialize(EngineConfig(
  modelPath: '/path/to/model.litertlm', // path on device
  backend: BackendType.gpu,
  visionBackend: BackendType.gpu, // required for image input
  cacheDir: '/path/to/cache',     // optional: speeds up subsequent loads
));
```

The model file must already be present on the device. Use your preferred
download/asset mechanism to place it before calling `initialize`.

### 2. Create a conversation

```dart
final conversation = await engine.createConversation(
  ConversationConfig(
    systemInstruction: 'You are a helpful assistant. Be concise.',
    temperature: 0.7,
    topK: 40,
    maxOutputTokens: 1024,
  ),
);
```

### 3a. Blocking response

```dart
final response = await conversation.sendText('What is the capital of France?');
print(response); // "Paris"
```

### 3b. Streaming response

```dart
final buffer = StringBuffer();
await for (final chunk in conversation.sendTextStream('Tell me a short story.')) {
  buffer.write(chunk);
  setState(() => _text = buffer.toString());
}
```

### 3c. Multimodal (image + text)

```dart
// From a file path (e.g. from image_picker)
final description = await conversation.sendMessage([
  LiteRtContent.imageFile('/storage/emulated/0/DCIM/photo.jpg'),
  LiteRtContent.text('What objects can you see in this image?'),
]);

// From raw bytes (e.g. from network or assets)
final Uint8List imageBytes = ...;
await conversation.sendMessage([
  LiteRtContent.imageBytes(imageBytes),
  LiteRtContent.text('Describe this image.'),
]);
```

### 4. Cancel an ongoing stream

```dart
await conversation.cancel();
```

### 5. Tear down

```dart
await conversation.close();
await engine.close();
```

---

## API Reference

### `LiteRtLmEngine`

| Method | Description |
|---|---|
| `initialize(EngineConfig)` | Loads model into memory. Blocking — takes several seconds. |
| `createConversation([ConversationConfig])` | Creates a new stateful conversation. |
| `close()` | Releases model and native resources. |
| `isInitialized` | Whether the engine is ready. |

### `LiteRtLmConversation`

| Method | Description |
|---|---|
| `sendText(String)` | Blocking text inference. |
| `sendMessage(List<LiteRtContent>)` | Blocking multimodal inference. |
| `sendTextStream(String)` | Streaming text inference. Returns `Stream<String>`. |
| `sendMessageStream(List<LiteRtContent>)` | Streaming multimodal inference. |
| `cancel()` | Cancels ongoing inference. |
| `close()` | Releases conversation resources. |

### `LiteRtContent` (sealed)

| Factory | Description |
|---|---|
| `LiteRtContent.text(String)` | Plain text part. |
| `LiteRtContent.imageFile(String path)` | Image from absolute file path. |
| `LiteRtContent.imageBytes(Uint8List)` | Image from raw bytes. |
| `LiteRtContent.audioFile(String path)` | Audio from absolute file path. |
| `LiteRtContent.audioBytes(Uint8List)` | Audio from raw bytes. |

### `EngineConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modelPath` | `String` | required | Absolute path to `.litertlm` model file. |
| `backend` | `BackendType` | `gpu` | Primary inference backend. |
| `visionBackend` | `BackendType?` | null | Vision backend (required for images). |
| `audioBackend` | `BackendType?` | null | Audio backend (required for audio). |
| `cacheDir` | `String?` | null | Directory for caching compiled model artifacts. |
| `maxNumTokens` | `int?` | null | Context window size override. |

---

## Error handling

All methods throw [`LiteRtLmException`] on failure:

```dart
try {
  await engine.initialize(config);
} on LiteRtLmException catch (e) {
  print('${e.code}: ${e.message}');
}
```

---

## iOS

iOS is not yet supported. All calls will throw a `PlatformException` with code
`NOT_SUPPORTED`. iOS support will be added when the official LiteRT-LM Swift
SDK is available.
