import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert_lm/flutter_litert_lm.dart';

void main() {
  // ──────────────────────────────────────────────────────────────────────
  // BackendType
  // ──────────────────────────────────────────────────────────────────────

  group('BackendType', () {
    test('name values are lowercase strings', () {
      expect(BackendType.cpu.name, 'cpu');
      expect(BackendType.gpu.name, 'gpu');
      expect(BackendType.npu.name, 'npu');
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // EngineConfig
  // ──────────────────────────────────────────────────────────────────────

  group('EngineConfig', () {
    test('toMap contains required fields', () {
      final config = const EngineConfig(modelPath: '/model.litertlm');
      final map = config.toMap();

      expect(map['modelPath'], '/model.litertlm');
      expect(map['backend'], 'gpu'); // default
      expect(map.containsKey('visionBackend'), isFalse);
      expect(map.containsKey('audioBackend'), isFalse);
    });

    test('toMap includes optional fields when set', () {
      final config = const EngineConfig(
        modelPath: '/model.litertlm',
        backend: BackendType.cpu,
        visionBackend: BackendType.gpu,
        audioBackend: BackendType.npu,
        cacheDir: '/cache',
        maxNumTokens: 2048,
      );
      final map = config.toMap();

      expect(map['backend'], 'cpu');
      expect(map['visionBackend'], 'gpu');
      expect(map['audioBackend'], 'npu');
      expect(map['cacheDir'], '/cache');
      expect(map['maxNumTokens'], 2048);
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // ConversationConfig
  // ──────────────────────────────────────────────────────────────────────

  group('ConversationConfig', () {
    test('toMap is empty with no options', () {
      final config = const ConversationConfig();
      expect(config.toMap(), isEmpty);
    });

    test('toMap includes only non-null fields', () {
      final config = const ConversationConfig(
        systemInstruction: 'Be concise.',
        topK: 40,
        topP: 0.9,
        temperature: 0.7,
        maxOutputTokens: 512,
      );
      final map = config.toMap();

      expect(map['systemInstruction'], 'Be concise.');
      expect(map['topK'], 40);
      expect(map['topP'], closeTo(0.9, 0.001));
      expect(map['temperature'], closeTo(0.7, 0.001));
      expect(map['maxOutputTokens'], 512);
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtContent
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtContent', () {
    test('text serialises correctly', () {
      final content = const LiteRtContent.text('Hello world');
      expect(content.toMap(), {'type': 'text', 'text': 'Hello world'});
    });

    test('imageFile serialises correctly', () {
      final content = const LiteRtContent.imageFile('/path/to/image.jpg');
      expect(content.toMap(), {'type': 'imageFile', 'path': '/path/to/image.jpg'});
    });

    test('imageBytes serialises correctly', () {
      final bytes = Uint8List.fromList([1, 2, 3]);
      final content = LiteRtContent.imageBytes(bytes);
      final map = content.toMap();
      expect(map['type'], 'imageBytes');
      expect(map['data'], bytes);
    });

    test('audioFile serialises correctly', () {
      final content = const LiteRtContent.audioFile('/path/to/audio.wav');
      expect(content.toMap(), {'type': 'audioFile', 'path': '/path/to/audio.wav'});
    });

    test('audioBytes serialises correctly', () {
      final bytes = Uint8List.fromList([4, 5, 6]);
      final content = LiteRtContent.audioBytes(bytes);
      final map = content.toMap();
      expect(map['type'], 'audioBytes');
      expect(map['data'], bytes);
    });

    test('pattern matching works on all subtypes', () {
      final List<LiteRtContent> parts = [
        const LiteRtContent.text('Hi'),
        const LiteRtContent.imageFile('/img.jpg'),
        LiteRtContent.imageBytes(Uint8List(0)),
        const LiteRtContent.audioFile('/audio.wav'),
        LiteRtContent.audioBytes(Uint8List(0)),
        const LiteRtContent.toolResponse('myTool', '{"result":1}'),
      ];

      final types = parts.map((p) => switch (p) {
            LiteRtTextContent()         => 'text',
            LiteRtImageFileContent()    => 'imageFile',
            LiteRtImageBytesContent()   => 'imageBytes',
            LiteRtAudioFileContent()    => 'audioFile',
            LiteRtAudioBytesContent()   => 'audioBytes',
            LiteRtToolResponseContent() => 'toolResponse',
          }).toList();

      expect(types, [
        'text', 'imageFile', 'imageBytes', 'audioFile', 'audioBytes', 'toolResponse',
      ]);
    });

    test('toolResponse serialises correctly', () {
      const content = LiteRtContent.toolResponse('weather', '{"temp":22}');
      expect(content.toMap(), {
        'type': 'toolResponse',
        'toolName': 'weather',
        'responseJson': '{"temp":22}',
      });
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtMessage
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtMessage', () {
    test('user() convenience constructor', () {
      final msg = LiteRtMessage.user('Hello');
      expect(msg.role, 'user');
      expect(msg.parts.length, 1);
      expect(msg.parts.first, isA<LiteRtTextContent>());
    });

    test('model() convenience constructor', () {
      final msg = LiteRtMessage.model('Paris.');
      expect(msg.role, 'model');
    });

    test('toMap serialises parts list', () {
      final msg = const LiteRtMessage(
        role: 'user',
        parts: [
          LiteRtContent.text('Describe this:'),
          LiteRtContent.imageFile('/photo.jpg'),
        ],
      );
      final map = msg.toMap();
      expect(map['role'], 'user');
      final parts = map['parts'] as List;
      expect(parts.length, 2);
      expect(parts[0]['type'], 'text');
      expect(parts[1]['type'], 'imageFile');
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtLmException
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtLmException', () {
    test('toString includes message', () {
      const e = LiteRtLmException('something went wrong');
      expect(e.toString(), contains('something went wrong'));
    });

    test('toString includes code when present', () {
      const e = LiteRtLmException('failed', code: 'ENGINE_INIT_FAILED');
      expect(e.toString(), contains('ENGINE_INIT_FAILED'));
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtToolDeclaration
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtToolDeclaration', () {
    test('toOpenApiJson encodes correct structure', () {
      const decl = LiteRtToolDeclaration(
        name: 'getCurrentWeather',
        description: 'Get weather for a city.',
        parameters: {
          'type': 'object',
          'properties': {
            'city': {'type': 'string'},
          },
          'required': ['city'],
        },
      );
      final json = decl.toOpenApiJson();
      expect(json, contains('getCurrentWeather'));
      expect(json, contains('Get weather for a city.'));
      expect(json, contains('city'));
    });

    test('toMap includes parametersJson key', () {
      const decl = LiteRtToolDeclaration(
        name: 'add',
        description: 'Add numbers.',
        parameters: {'type': 'object', 'properties': {}},
      );
      final map = decl.toMap();
      expect(map['name'], 'add');
      expect(map['description'], 'Add numbers.');
      expect(map.containsKey('parametersJson'), isTrue);
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtToolCall
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtToolCall', () {
    test('arguments getter decodes JSON', () {
      const call = LiteRtToolCall(
        name: 'getCurrentWeather',
        argumentsJson: '{"city":"London","unit":"celsius"}',
      );
      expect(call.arguments['city'], 'London');
      expect(call.arguments['unit'], 'celsius');
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // LiteRtMessageResponse
  // ──────────────────────────────────────────────────────────────────────

  group('LiteRtMessageResponse', () {
    test('fromMap decodes text-only response', () {
      final response = LiteRtMessageResponse.fromMap({
        'text': 'Paris',
        'toolCalls': <Object?>[],
      });
      expect(response.text, 'Paris');
      expect(response.hasToolCalls, isFalse);
    });

    test('fromMap decodes tool-call response', () {
      final response = LiteRtMessageResponse.fromMap({
        'text': '',
        'toolCalls': [
          {'name': 'getCurrentWeather', 'argumentsJson': '{"city":"London"}'},
        ],
      });
      expect(response.hasToolCalls, isTrue);
      expect(response.toolCalls.first.name, 'getCurrentWeather');
      expect(response.toolCalls.first.arguments['city'], 'London');
    });
  });

  // ──────────────────────────────────────────────────────────────────────
  // ConversationConfig with tools
  // ──────────────────────────────────────────────────────────────────────

  group('ConversationConfig (tools)', () {
    test('toMap includes tools and automaticToolCalling=false when tools provided', () {
      final config = const ConversationConfig(
        tools: [
          LiteRtToolDeclaration(
            name: 'getDate',
            description: 'Get current date.',
            parameters: {'type': 'object', 'properties': {}},
          ),
        ],
      );
      final map = config.toMap();
      expect(map.containsKey('tools'), isTrue);
      expect((map['tools'] as List).length, 1);
      expect(map['automaticToolCalling'], isFalse);
    });

    test('toMap does not include tools when list is empty', () {
      const config = ConversationConfig();
      expect(config.toMap().containsKey('tools'), isFalse);
      expect(config.toMap().containsKey('automaticToolCalling'), isFalse);
    });
  });
}
