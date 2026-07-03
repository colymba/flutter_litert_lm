package dev.colymba.flutter_litert_lm

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel

/**
 * FlutterLiteRtLmPlugin
 *
 * Main plugin entry point. Registers the primary [MethodChannel] and delegates
 * calls to [EngineHandler] and [ConversationHandler].
 *
 * Channel name: `flutter_litert_lm`
 */
class FlutterLiteRtLmPlugin : FlutterPlugin, MethodCallHandler {

    private lateinit var channel: MethodChannel
    private lateinit var messenger: BinaryMessenger

    private lateinit var engineHandler: EngineHandler
    private lateinit var conversationHandler: ConversationHandler
    private lateinit var embedderHandler: EmbedderHandler

    // Coroutine scope for all plugin work. Cancelled on detach.
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    // ──────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ──────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        messenger = binding.binaryMessenger

        channel = MethodChannel(messenger, "flutter_litert_lm")
        channel.setMethodCallHandler(this)

        engineHandler = EngineHandler(scope)
        conversationHandler = ConversationHandler(scope, messenger).also {
            it.engineHandler = engineHandler
        }
        embedderHandler = EmbedderHandler(scope)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)

        // Release all native resources on detach so the app doesn't leak.
        engineHandler.closeAll()
        conversationHandler.closeAll()
        embedderHandler.closeAll()

        scope.cancel()
    }

    // ──────────────────────────────────────────────────────────────────────
    // MethodCallHandler
    // ──────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: Result) {
        when {
            call.method.startsWith("engine/") ->
                engineHandler.handle(call, result)

            call.method.startsWith("conversation/") ->
                conversationHandler.handle(call, result)

            call.method.startsWith("embedder/") ->
                embedderHandler.handle(call, result)

            else ->
                result.notImplemented()
        }
    }
}
