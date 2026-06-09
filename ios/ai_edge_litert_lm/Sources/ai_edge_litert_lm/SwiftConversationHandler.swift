import Flutter
import LiteRTLM
import Foundation

class SwiftConversationHandler {
    private struct ConversationEntry {
        let conversation: Conversation
        let engine: Engine
    }
    
    private var conversations: [String: ConversationEntry] = [:]
    private var streamHandlers: [String: SwiftStreamHandler] = [:]
    private var counter = 0
    private let lock = NSLock()
    
    private let messenger: FlutterBinaryMessenger
    weak var engineHandler: SwiftEngineHandler?
    
    init(messenger: FlutterBinaryMessenger) {
        self.messenger = messenger
    }
    
    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "conversation/create":
            create(call, result: result)
        case "conversation/sendMessage":
            sendMessage(call, result: result)
        case "conversation/sendMessageStream":
            sendMessageStream(call, result: result)
        case "conversation/cancel":
            cancel(call, result: result)
        case "conversation/close":
            close(call, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    func closeAll() {
        lock.lock()
        let activeConversations = conversations
        let activeHandlers = streamHandlers
        conversations.removeAll()
        streamHandlers.removeAll()
        lock.unlock()
        
        for entry in activeConversations.values {
            try? entry.conversation.close()
        }
        for (convId, _) in activeHandlers {
            let streamChannelName = "flutter_litert_lm/stream/\(convId)"
            let channel = FlutterEventChannel(name: streamChannelName, binaryMessenger: messenger)
            channel.setStreamHandler(nil)
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // create
    // ──────────────────────────────────────────────────────────────────────
    
    private func create(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let engineId = args["engineId"] as? String else {
            result(FlutterError(code: "INVALID_ARG", message: "engineId is required", details: nil))
            return
        }
        
        guard let engine = engineHandler?.getEngine(engineId) else {
            result(FlutterError(code: "NOT_FOUND", message: "No engine found for id: \(engineId)", details: nil))
            return
        }
        
        let systemInstruction = args["systemInstruction"] as? String
        let topK = args["topK"] as? Int
        let topP = args["topP"] as? Double
        let temperature = args["temperature"] as? Double
        let maxOutputTokens = args["maxOutputTokens"] as? Int
        let automaticToolCalling = args["automaticToolCalling"] as? Bool ?? true
        let toolsArg = args["tools"] as? [[String: Any]]
        
        Task {
            do {
                let samplerConfig = try SamplerConfig(
                    topK: topK ?? 40,
                    topP: topP ?? 0.95,
                    temperature: temperature ?? 0.8
                )
                
                var toolInstances: [OpenApiTool] = []
                if let toolsArg = toolsArg {
                    for toolMap in toolsArg {
                        if let tName = toolMap["name"] as? String,
                           let tDesc = toolMap["description"] as? String,
                           let tParamsJson = toolMap["parametersJson"] as? String {
                            toolInstances.append(DartOpenApiTool(name: tName, description: tDesc, parametersJson: tParamsJson))
                        }
                    }
                }
                
                let config = ConversationConfig(
                    systemMessage: systemInstruction.map { Message(role: .system, content: $0) },
                    samplerConfig: samplerConfig,
                    tools: toolInstances,
                    automaticToolCalling: automaticToolCalling
                )
                
                let conversation = try await engine.createConversation(with: config)
                
                lock.lock()
                counter += 1
                let conversationId = "conversation_\(counter)"
                conversations[conversationId] = ConversationEntry(conversation: conversation, engine: engine)
                lock.unlock()
                
                result(conversationId)
            } catch {
                result(FlutterError(
                    code: "CONVERSATION_CREATE_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // sendMessage (blocking)
    // ──────────────────────────────────────────────────────────────────────
    
    private func sendMessage(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let conversationId = args["conversationId"] as? String,
              let partsArg = args["parts"] as? [[String: Any]] else {
            result(FlutterError(code: "INVALID_ARG", message: "conversationId and parts are required", details: nil))
            return
        }
        
        let roleStr = args["role"] as? String
        
        lock.lock()
        let entry = conversations[conversationId]
        lock.unlock()
        
        guard let entry = entry else {
            result(FlutterError(code: "NOT_FOUND", message: "No conversation found for id: \(conversationId)", details: nil))
            return
        }
        
        Task {
            do {
                let contents = try buildContents(partsArg)
                let role = parseRole(roleStr)
                let message = Message(role: role, contents: contents)
                
                let responseMessage = try await entry.conversation.sendMessage(message)
                
                result(messageToMap(responseMessage))
            } catch let error as FlutterError {
                result(error)
            } catch {
                result(FlutterError(
                    code: "SEND_MESSAGE_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // sendMessageStream (EventChannel)
    // ──────────────────────────────────────────────────────────────────────
    
    private func sendMessageStream(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let conversationId = args["conversationId"] as? String,
              let partsArg = args["parts"] as? [[String: Any]] else {
            result(FlutterError(code: "INVALID_ARG", message: "conversationId and parts are required", details: nil))
            return
        }
        
        lock.lock()
        let entry = conversations[conversationId]
        lock.unlock()
        
        guard let entry = entry else {
            result(FlutterError(code: "NOT_FOUND", message: "No conversation found for id: \(conversationId)", details: nil))
            return
        }
        
        let streamChannelName = "flutter_litert_lm/stream/\(conversationId)"
        let eventChannel = FlutterEventChannel(name: streamChannelName, binaryMessenger: messenger)
        
        lock.lock()
        if streamHandlers[conversationId] != nil {
            lock.unlock()
            cleanupStreamChannel(conversationId)
            lock.lock()
        }
        lock.unlock()
        
        let handler = SwiftStreamHandler { [weak self] _, events in
            guard let self = self else { return }
            
            Task {
                do {
                    let contents = try self.buildContents(partsArg)
                    let message = Message(role: .user, contents: contents)
                    
                    let stream = try await entry.conversation.sendMessageStream(message)
                    var lastMessage: Message?
                    
                    for try await chunkMessage in stream {
                        lastMessage = chunkMessage
                        var chunkText = ""
                        for content in chunkMessage.contents {
                            if case .text(let t) = content {
                                chunkText += t
                            }
                        }
                        if !chunkText.isEmpty {
                            events(chunkText)
                        }
                    }
                    
                    if let lastMessage = lastMessage, !lastMessage.toolCalls.isEmpty {
                        let calls = lastMessage.toolCalls.map { tc -> [String: Any] in
                            return [
                                "name": tc.name,
                                "argumentsJson": tc.arguments
                            ]
                        }
                        events(["_type": "toolCalls", "calls": calls])
                    }
                    
                    events(FlutterEndOfEventStream)
                } catch {
                    events(FlutterError(
                        code: "STREAM_ERROR",
                        message: error.localizedDescription,
                        details: String(describing: error)
                    ))
                }
            }
        } onCancel: { [weak self] _ in
            guard let self = self else { return }
            self.cleanupStreamChannel(conversationId)
        }
        
        eventChannel.setStreamHandler(handler)
        
        lock.lock()
        streamHandlers[conversationId] = handler
        lock.unlock()
        
        result(streamChannelName)
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // cancel
    // ──────────────────────────────────────────────────────────────────────
    
    private func cancel(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let conversationId = args["conversationId"] as? String else {
            result(FlutterError(code: "INVALID_ARG", message: "conversationId is required", details: nil))
            return
        }
        
        lock.lock()
        let entry = conversations[conversationId]
        lock.unlock()
        
        guard let entry = entry else {
            result(nil) // Idempotent cancel
            return
        }
        
        Task {
            do {
                try entry.conversation.cancelProcess()
                result(nil)
            } catch {
                result(FlutterError(
                    code: "CANCEL_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // close
    // ──────────────────────────────────────────────────────────────────────
    
    private func close(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let conversationId = args["conversationId"] as? String else {
            result(FlutterError(code: "INVALID_ARG", message: "conversationId is required", details: nil))
            return
        }
        
        lock.lock()
        let entry = conversations.removeValue(forKey: conversationId)
        lock.unlock()
        
        cleanupStreamChannel(conversationId)
        
        guard let entry = entry else {
            result(nil) // Idempotent close
            return
        }
        
        Task {
            do {
                try entry.conversation.close()
                result(nil)
            } catch {
                result(FlutterError(
                    code: "CONVERSATION_CLOSE_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────
    
    private func buildContents(_ parts: [[String: Any]]) throws -> [Content] {
        return try parts.map { part in
            guard let type = part["type"] as? String else {
                throw FlutterError(code: "INVALID_ARG", message: "part type is required", details: nil)
            }
            switch type {
            case "text":
                guard let text = part["text"] as? String else {
                    throw FlutterError(code: "INVALID_ARG", message: "text is required for text part", details: nil)
                }
                return .text(text)
            case "imageFile":
                guard let path = part["path"] as? String else {
                    throw FlutterError(code: "INVALID_ARG", message: "path is required for imageFile part", details: nil)
                }
                return .imageFile(path)
            case "imageBytes":
                let data: Data
                if let flutterData = part["data"] as? FlutterStandardTypedData {
                    data = flutterData.data
                } else if let rawData = part["data"] as? Data {
                    data = rawData
                } else {
                    throw FlutterError(code: "INVALID_ARG", message: "data is required for imageBytes part", details: nil)
                }
                return .imageBytes(data)
            case "audioFile":
                guard let path = part["path"] as? String else {
                    throw FlutterError(code: "INVALID_ARG", message: "path is required for audioFile part", details: nil)
                }
                return .audioFile(path)
            case "audioBytes":
                let data: Data
                if let flutterData = part["data"] as? FlutterStandardTypedData {
                    data = flutterData.data
                } else if let rawData = part["data"] as? Data {
                    data = rawData
                } else {
                    throw FlutterError(code: "INVALID_ARG", message: "data is required for audioBytes part", details: nil)
                }
                return .audioBytes(data)
            case "toolResponse":
                guard let toolName = part["toolName"] as? String,
                      let responseJson = part["responseJson"] as? String else {
                    throw FlutterError(code: "INVALID_ARG", message: "toolName and responseJson are required for toolResponse part", details: nil)
                }
                return .toolResponse(name: toolName, responseJson: responseJson)
            default:
                throw FlutterError(code: "INVALID_ARG", message: "Unknown content type: \(type)", details: nil)
            }
        }
    }
    
    private func parseRole(_ value: String?) -> Message.Role {
        switch value?.lowercased() {
        case "user": return .user
        case "model": return .model
        case "system": return .system
        case "tool": return .tool
        default: return .user
        }
    }
    
    private func messageToMap(_ message: Message) -> [String: Any] {
        var text = ""
        for content in message.contents {
            if case .text(let t) = content {
                text += t
            }
        }
        
        let toolCalls = message.toolCalls.map { tc -> [String: Any] in
            return [
                "name": tc.name,
                "argumentsJson": tc.arguments
            ]
        }
        
        return [
            "text": text,
            "toolCalls": toolCalls
        ]
    }
    
    private func cleanupStreamChannel(_ conversationId: String) {
        lock.lock()
        let handler = streamHandlers.removeValue(forKey: conversationId)
        lock.unlock()
        
        if handler != nil {
            let streamChannelName = "flutter_litert_lm/stream/\(conversationId)"
            let channel = FlutterEventChannel(name: streamChannelName, binaryMessenger: messenger)
            channel.setStreamHandler(nil)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// SwiftStreamHandler Helper
// ──────────────────────────────────────────────────────────────────────

class SwiftStreamHandler: NSObject, FlutterStreamHandler {
    private let onListenBlock: (Any?, @escaping FlutterEventSink) -> Void
    private let onCancelBlock: (Any?) -> Void
    
    init(onListen: @escaping (Any?, @escaping FlutterEventSink) -> Void, onCancel: @escaping (Any?) -> Void) {
        self.onListenBlock = onListen
        self.onCancelBlock = onCancel
    }
    
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        onListenBlock(arguments, events)
        return nil
    }
    
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        onCancelBlock(arguments)
        return nil
    }
}

// ──────────────────────────────────────────────────────────────────────
// DartOpenApiTool Bridge
// ──────────────────────────────────────────────────────────────────────

class DartOpenApiTool: NSObject, OpenApiTool {
    let name: String
    let toolDescription: String
    let parametersJson: String
    
    init(name: String, description: String, parametersJson: String) {
        self.name = name
        self.toolDescription = description
        self.parametersJson = parametersJson
        super.init()
    }
    
    func getToolDescriptionJsonString() -> String {
        return "{\"name\":\"\(name)\",\"description\":\"\(toolDescription)\",\"parameters\":\(parametersJson)}"
    }
    
    func execute(paramsJsonString: String) -> String {
        fatalError("DartOpenApiTool.execute() should not be called — tool execution is handled in Dart.")
    }
}
