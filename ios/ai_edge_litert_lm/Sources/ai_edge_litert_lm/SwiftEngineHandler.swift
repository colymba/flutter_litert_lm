import Flutter
import LiteRTLM
import Foundation

class SwiftEngineHandler {
    private var engines: [String: Engine] = [:]
    private var counter = 0
    private let lock = NSLock()
    
    func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "engine/initialize":
            initialize(call, result: result)
        case "engine/close":
            close(call, result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }
    
    func getEngine(_ engineId: String) -> Engine? {
        lock.lock()
        defer { lock.unlock() }
        return engines[engineId]
    }
    
    func closeAll() {
        lock.lock()
        let activeEngines = engines
        engines.removeAll()
        lock.unlock()
        
        for engine in activeEngines.values {
            try? engine.close()
        }
    }
    
    private func initialize(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let modelPath = args["modelPath"] as? String else {
            result(FlutterError(code: "INVALID_ARG", message: "modelPath is required", details: nil))
            return
        }
        
        let backendStr = args["backend"] as? String ?? "gpu"
        let visionBackendStr = args["visionBackend"] as? String
        let audioBackendStr = args["audioBackend"] as? String
        let cacheDir = args["cacheDir"] as? String
        let maxNumTokens = args["maxNumTokens"] as? Int
        
        if let cacheDir = cacheDir {
            try? FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
        }
        
        Task {
            do {
                let config = try EngineConfig(
                    modelPath: modelPath,
                    backend: parseBackend(backendStr),
                    visionBackend: visionBackendStr.map { parseBackend($0) },
                    audioBackend: audioBackendStr.map { parseBackend($0) },
                    maxNumTokens: maxNumTokens,
                    cacheDir: cacheDir
                )
                
                let engine = Engine(engineConfig: config)
                try await engine.initialize()
                
                lock.lock()
                counter += 1
                let engineId = "engine_\(counter)"
                engines[engineId] = engine
                lock.unlock()
                
                result(engineId)
            } catch {
                result(FlutterError(
                    code: "ENGINE_INIT_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    private func close(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any],
              let engineId = args["engineId"] as? String else {
            result(FlutterError(code: "INVALID_ARG", message: "engineId is required", details: nil))
            return
        }
        
        lock.lock()
        let engine = engines.removeValue(forKey: engineId)
        lock.unlock()
        
        guard let engine = engine else {
            result(FlutterError(code: "NOT_FOUND", message: "No engine found for id: \(engineId)", details: nil))
            return
        }
        
        Task {
            do {
                try engine.close()
                result(nil)
            } catch {
                result(FlutterError(
                    code: "ENGINE_CLOSE_FAILED",
                    message: error.localizedDescription,
                    details: String(describing: error)
                ))
            }
        }
    }
    
    private func parseBackend(_ value: String) -> Backend {
        switch value.lowercased() {
        case "cpu": return .cpu
        case "gpu": return .gpu
        case "npu": return .npu
        default: return .gpu
        }
    }
}
