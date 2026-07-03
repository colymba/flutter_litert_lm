import Flutter
import UIKit

public class FlutterLiteRtLmPlugin: NSObject, FlutterPlugin {
    private let engineHandler = SwiftEngineHandler()
    private let conversationHandler: SwiftConversationHandler
    
    init(messenger: FlutterBinaryMessenger) {
        self.conversationHandler = SwiftConversationHandler(messenger: messenger)
        self.conversationHandler.engineHandler = engineHandler
        super.init()
    }
    
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "flutter_litert_lm", binaryMessenger: registrar.messenger())
        let instance = FlutterLiteRtLmPlugin(messenger: registrar.messenger())
        registrar.addMethodCallDelegate(instance, channel: channel)
    }
    
    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        if call.method.hasPrefix("engine/") {
            engineHandler.handle(call, result: result)
        } else if call.method.hasPrefix("conversation/") {
            conversationHandler.handle(call, result: result)
        } else if call.method.hasPrefix("embedder/") {
            // The embedder runs on the LiteRT interpreter, which is not part
            // of the LiteRT-LM Swift SDK. Android-only for now.
            result(FlutterError(
                code: "NOT_SUPPORTED",
                message: "The embedder API is not yet supported on iOS.",
                details: nil
            ))
        } else {
            result(FlutterMethodNotImplemented)
        }
    }
    
    public func detachFromEngine(for registrar: FlutterPluginRegistrar) {
        engineHandler.closeAll()
        conversationHandler.closeAll()
    }
}
