/// Flutter LiteRT-LM
///
/// A Flutter plugin for on-device LLM inference using Google's LiteRT-LM
/// framework. Wraps the high-level Conversation API for stateful, multi-turn
/// chat with full multimodal (text + image + audio) support.
///
/// ## Quick start
///
/// ```dart
/// import 'package:flutter_litert_lm/flutter_litert_lm.dart';
///
/// // 1. Create and initialise the engine (heavyweight — do once on startup)
/// final engine = LiteRtLmEngine();
/// await engine.initialize(EngineConfig(
///   modelPath: '/path/to/model.litertlm',
///   backend: BackendType.gpu,
///   visionBackend: BackendType.gpu, // required for image input
/// ));
///
/// // 2. Create a conversation
/// final conversation = await engine.createConversation(
///   ConversationConfig(systemInstruction: 'Be concise.'),
/// );
///
/// // 3a. Blocking text response
/// final answer = await conversation.sendText('What is the capital of France?');
///
/// // 3b. Streaming response
/// await for (final chunk in conversation.sendTextStream('Tell me a story')) {
///   print(chunk); // prints tokens as they arrive
/// }
///
/// // 3c. Multimodal (image + text)
/// final description = await conversation.sendMessage([
///   LiteRtContent.imageFile('/path/to/photo.jpg'),
///   LiteRtContent.text('Describe this image.'),
/// ]);
///
/// // 4. Tear down
/// await conversation.close();
/// await engine.close();
/// ```
library;

export 'src/litert_lm_conversation.dart' show LiteRtLmConversation;
export 'src/litert_lm_engine.dart' show LiteRtLmEngine;
export 'src/models.dart'
    show
        BackendType,
        ConversationConfig,
        EngineConfig,
        LiteRtAudioBytesContent,
        LiteRtAudioFileContent,
        LiteRtContent,
        LiteRtImageBytesContent,
        LiteRtImageFileContent,
        LiteRtLmException,
        LiteRtMessage,
        LiteRtMessageResponse,
        LiteRtTextContent,
        LiteRtToolCall,
        LiteRtToolDeclaration,
        LiteRtToolResponseContent;
