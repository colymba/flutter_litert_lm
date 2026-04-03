package dev.colymba.flutter_litert_lm

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * EngineHandler
 *
 * Handles all `engine/...` MethodChannel calls.
 *
 * Engine instances are identified by a string ID returned to Dart. Dart passes
 * this ID back for `createConversation` and `engine/close`.
 */
class EngineHandler(private val scope: CoroutineScope) {

    private val engines = ConcurrentHashMap<String, Engine>()
    private val counter = AtomicInteger(0)

    // ──────────────────────────────────────────────────────────────────────
    // Dispatch
    // ──────────────────────────────────────────────────────────────────────

    fun handle(call: MethodCall, result: Result) {
        when (call.method) {
            "engine/initialize" -> initialize(call, result)
            "engine/close"      -> close(call, result)
            else                -> result.notImplemented()
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // initialize
    // ──────────────────────────────────────────────────────────────────────

    private fun initialize(call: MethodCall, result: Result) {
        val modelPath: String = call.argument("modelPath")
            ?: return result.error("INVALID_ARG", "modelPath is required", null)

        val backendStr: String = call.argument<String>("backend") ?: "gpu"
        val visionBackendStr: String? = call.argument("visionBackend")
        val audioBackendStr: String? = call.argument("audioBackend")
        val cacheDir: String? = call.argument("cacheDir")
        cacheDir?.let { java.io.File(it).mkdirs() }

        val maxNumTokens: Int? = call.argument("maxNumTokens")

        scope.launch {
            try {
                val engine = withContext(Dispatchers.IO) {
                    val config = EngineConfig(
                        modelPath = modelPath,
                        backend = parseBackend(backendStr),
                        visionBackend = visionBackendStr?.let { parseBackend(it) },
                        audioBackend = audioBackendStr?.let { parseBackend(it) },
                        cacheDir = cacheDir,
                        maxNumTokens = maxNumTokens,
                    )
                    val engine = Engine(config)
                    engine.initialize()
                    engine
                }

                val engineId = "engine_${counter.incrementAndGet()}"
                engines[engineId] = engine
                result.success(engineId)

            } catch (e: Exception) {
                result.error(
                    "ENGINE_INIT_FAILED",
                    e.message ?: "Engine initialisation failed",
                    e.javaClass.simpleName
                )
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // close
    // ──────────────────────────────────────────────────────────────────────

    private fun close(call: MethodCall, result: Result) {
        val engineId: String = call.argument("engineId")
            ?: return result.error("INVALID_ARG", "engineId is required", null)

        val engine = engines.remove(engineId)
            ?: return result.error("NOT_FOUND", "No engine found for id: $engineId", null)

        scope.launch {
            try {
                withContext(Dispatchers.IO) { engine.close() }
                result.success(null)
            } catch (e: Exception) {
                result.error("ENGINE_CLOSE_FAILED", e.message, null)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ──────────────────────────────────────────────────────────────────────

    /** Returns the Engine for [engineId], or null if not found. */
    fun getEngine(engineId: String): Engine? = engines[engineId]

    /** Closes all engines. Called on plugin detach. */
    fun closeAll() {
        engines.values.forEach { engine ->
            try { engine.close() } catch (_: Exception) {}
        }
        engines.clear()
    }

    // ──────────────────────────────────────────────────────────────────────
    // Backend parsing
    // ──────────────────────────────────────────────────────────────────────

    private fun parseBackend(value: String): Backend = when (value.lowercase()) {
        "cpu" -> Backend.CPU()
        "gpu" -> Backend.GPU()
        "npu" -> Backend.NPU()
        else  -> throw IllegalArgumentException("Unknown backend: $value. Use 'cpu', 'gpu', or 'npu'.")
    }
}
