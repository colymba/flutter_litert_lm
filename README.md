# flutter_litert_lm

A Flutter plugin for on-device LLM inference using Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) framework.

| Platform | Status |
|---|---|
| Android | ✅ Supported via official `litertlm-android` Kotlin SDK |
| iOS | 🚧 Stub — pending official Swift SDK |

---

## Features

- **Stateful multi-turn conversations** — tracks history automatically.
- **Streaming responses** — receive tokens as they are generated via a `Stream<LiteRtMessageResponse>`.
- **Multimodal inputs** — text, images (file path or raw bytes), and audio (file path or raw bytes) in the same message.
- **Tool calling** — register Dart functions the model can invoke; intercept calls and feed results back.
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
print(response.text); // "Paris"
```

### 3b. Streaming response

```dart
final buffer = StringBuffer();
await for (final event in conversation.sendTextStream('Tell me a short story.')) {
  buffer.write(event.text);
  setState(() => _text = buffer.toString());
}
```

### 3c. Multimodal (image + text)

```dart
// From a file path (e.g. from image_picker)
final response = await conversation.sendMessage([
  LiteRtContent.imageFile('/storage/emulated/0/DCIM/photo.jpg'),
  LiteRtContent.text('What objects can you see in this image?'),
]);

// From raw bytes (e.g. from network or assets)
final Uint8List imageBytes = ...;
final response2 = await conversation.sendMessage([
  LiteRtContent.imageBytes(imageBytes),
  LiteRtContent.text('Describe this image.'),
]);
```

### 3d. Tool calling

Tool calling lets the model invoke Dart-defined functions to fetch information
or perform actions before generating its final response. This requires a
[tool-capable model](https://huggingface.co/google/functiongemma-270m-it).

#### Register tools when creating the conversation

```dart
final weatherDeclaration = LiteRtToolDeclaration(
  name: 'getCurrentWeather',
  description: 'Returns the current weather conditions for a given city.',
  parameters: {
    'type': 'object',
    'properties': {
      'city': {
        'type': 'string',
        'description': 'The city name, e.g. London',
      },
      'unit': {
        'type': 'string',
        'enum': ['celsius', 'fahrenheit'],
        'description': 'Temperature unit. Default: celsius',
      },
    },
    'required': ['city'],
  },
);

final conversation = await engine.createConversation(
  ConversationConfig(
    systemInstruction: 'You are a helpful assistant.',
    tools: [weatherDeclaration],
  ),
);
```

#### Blocking round-trip

```dart
import 'dart:convert';

Future<void> chat(LiteRtLmConversation conversation) async {
  final response = await conversation.sendText("What's the weather like in London?");

  if (response.hasToolCalls) {
    // Build tool responses for every call the model requested.
    final toolResponseParts = response.toolCalls.map((call) {
      final args = call.arguments; // Map<String, dynamic>
      // Execute tool locally and encode result as JSON.
      final result = switch (call.name) {
        'getCurrentWeather' => {
          'temperature': 18,
          'unit': args['unit'] ?? 'celsius',
          'condition': 'Partly cloudy',
        },
        _ => {'error': 'Unknown tool'},
      };
      return LiteRtContent.toolResponse(call.name, jsonEncode(result));
    }).toList();

    // Feed results back; get the model's final answer.
    final finalResponse = await conversation.sendToolResponses(toolResponseParts);
    print(finalResponse.text); // e.g. "The weather in London is 18°C and partly cloudy."
  } else {
    print(response.text);
  }
}
```

#### Streaming round-trip

```dart
Stream<String> streamWithToolCalls(LiteRtLmConversation conversation) async* {
  await for (final event in conversation.sendTextStream("What's the weather in Tokyo?")) {
    if (event.hasToolCalls) {
      // Terminal tool-call event — execute tools and continue.
      final toolResponseParts = event.toolCalls.map((call) {
        final result = myExecuteTool(call.name, call.arguments);
        return LiteRtContent.toolResponse(call.name, jsonEncode(result));
      }).toList();

      final finalResponse = await conversation.sendToolResponses(toolResponseParts);
      yield finalResponse.text;
    } else {
      yield event.text; // stream each token to the UI
    }
  }
}
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
| `sendText(String)` | Blocking text inference. Returns `LiteRtMessageResponse`. |
| `sendMessage(List<LiteRtContent>)` | Blocking multimodal inference. Returns `LiteRtMessageResponse`. |
| `sendTextStream(String)` | Streaming text inference. Returns `Stream<LiteRtMessageResponse>`. |
| `sendMessageStream(List<LiteRtContent>)` | Streaming multimodal inference. |
| `sendToolResponses(List<LiteRtContent>)` | Sends tool results back and returns the model's final response. |
| `cancel()` | Cancels ongoing inference. |
| `close()` | Releases conversation resources. |

### `LiteRtMessageResponse`

| Property | Type | Description |
|---|---|---|
| `text` | `String` | The model's text output. May be empty when `hasToolCalls` is `true`. |
| `toolCalls` | `List<LiteRtToolCall>` | Tool invocations requested by the model. |
| `hasToolCalls` | `bool` | `true` when the model wants to call tools instead of returning text. |

### `LiteRtToolDeclaration`

| Parameter | Type | Description |
|---|---|---|
| `name` | `String` | Function name the model uses to invoke the tool. |
| `description` | `String` | Short description so the model knows when to use the tool. |
| `parameters` | `Map<String, dynamic>` | JSON Schema *object* for the function's arguments. |

### `LiteRtToolCall`

| Property | Type | Description |
|---|---|---|
| `name` | `String` | Name of the tool the model wants to call. |
| `argumentsJson` | `String` | Raw JSON string of the argument object. |
| `arguments` | `Map<String, dynamic>` | Convenience getter: decoded `argumentsJson`. |

### `LiteRtContent` (sealed)

| Factory | Description |
|---|---|
| `LiteRtContent.text(String)` | Plain text part. |
| `LiteRtContent.imageFile(String path)` | Image from absolute file path. |
| `LiteRtContent.imageBytes(Uint8List)` | Image from raw bytes. |
| `LiteRtContent.audioFile(String path)` | Audio from absolute file path. |
| `LiteRtContent.audioBytes(Uint8List)` | Audio from raw bytes. |
| `LiteRtContent.toolResponse(String name, String json)` | Tool execution result to return to the model. |

### `EngineConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `modelPath` | `String` | required | Absolute path to `.litertlm` model file. |
| `backend` | `BackendType` | `gpu` | Primary inference backend. |
| `visionBackend` | `BackendType?` | null | Vision backend (required for images). |
| `audioBackend` | `BackendType?` | null | Audio backend (required for audio). |
| `cacheDir` | `String?` | null | Directory for caching compiled model artifacts. |
| `maxNumTokens` | `int?` | null | Context window size override. |

### `ConversationConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `systemInstruction` | `String?` | null | System-level instruction prepended to every conversation. |
| `topK` | `int?` | null | Top-K sampling parameter. |
| `topP` | `double?` | null | Top-P (nucleus) sampling parameter. |
| `temperature` | `double?` | null | Sampling temperature. |
| `maxOutputTokens` | `int?` | null | Maximum tokens to generate per response. |
| `tools` | `List<LiteRtToolDeclaration>` | `[]` | Tools the model may invoke. Enables manual tool calling when non-empty. |

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
