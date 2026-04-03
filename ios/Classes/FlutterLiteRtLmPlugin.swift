import Flutter
import UIKit

/// iOS stub for FlutterLiteRtLmPlugin.
///
/// All method calls return a "NOT_SUPPORTED" PlatformException until the
/// official LiteRT-LM Swift SDK is available and integrated.
public class FlutterLiteRtLmPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "flutter_litert_lm",
            binaryMessenger: registrar.messenger()
        )
        let instance = FlutterLiteRtLmPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        result(FlutterError(
            code: "NOT_SUPPORTED",
            message: "LiteRT-LM is not yet supported on iOS. "
                + "iOS support will be added when the official Swift SDK is available.",
            details: nil
        ))
    }
}
