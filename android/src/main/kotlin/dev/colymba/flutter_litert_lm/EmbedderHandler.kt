package dev.colymba.flutter_litert_lm

import ai.djl.sentencepiece.SpTokenizer
import ai.djl.sentencepiece.SpVocabulary
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Paths
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.sqrt

/**
 * EmbedderHandler
 *
 * Handles all `embedder/...` MethodChannel calls.
 *
 * Runs an embedding encoder model (e.g. EmbeddingGemma `.tflite` from
 * `litert-community`) on the LiteRT interpreter, with sentencepiece
 * tokenization via DJL. Deliberately independent from [EngineHandler]:
 * embedders are small encoders that should not require loading a generation
 * model, and vice versa.
 *
 * Tensor contract (introspected at initialize time, not hard-coded):
 * - input 0: int32 [1, seqLen] token ids, zero-padded
 * - output 0: float32 [1, dim] embedding
 */
class EmbedderHandler(private val scope: CoroutineScope) {

    private class EmbedderInstance(
        val interpreter: Interpreter,
        val tokenizer: SpTokenizer,
        val vocabulary: SpVocabulary,
        val seqLen: Int,
        val dimension: Int,
        val normalize: Boolean,
    ) {
        // The TFLite interpreter is not thread-safe; serialize inference.
        val mutex = Mutex()

        fun close() {
            interpreter.close()
            tokenizer.close()
        }
    }

    private val embedders = ConcurrentHashMap<String, EmbedderInstance>()
    private val counter = AtomicInteger(0)

    // ──────────────────────────────────────────────────────────────────────
    // Dispatch
    // ──────────────────────────────────────────────────────────────────────

    fun handle(call: MethodCall, result: Result) {
        when (call.method) {
            "embedder/initialize" -> initialize(call, result)
            "embedder/embed"      -> embed(call, result)
            "embedder/embedBatch" -> embedBatch(call, result)
            "embedder/close"      -> close(call, result)
            else                  -> result.notImplemented()
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // initialize
    // ──────────────────────────────────────────────────────────────────────

    private fun initialize(call: MethodCall, result: Result) {
        val modelPath: String = call.argument("modelPath")
            ?: return result.error("INVALID_ARG", "modelPath is required", null)
        val tokenizerPath: String = call.argument("tokenizerPath")
            ?: return result.error("INVALID_ARG", "tokenizerPath is required", null)
        val normalize: Boolean = call.argument<Boolean>("normalize") ?: true
        // Backend: CPU (XNNPack) only for now. GPU delegates add little for
        // small encoders and would compete with the generation engine.

        scope.launch {
            try {
                val instance = withContext(Dispatchers.IO) {
                    val modelFile = File(modelPath)
                    require(modelFile.exists()) { "Model file not found: $modelPath" }
                    require(File(tokenizerPath).exists()) {
                        "Tokenizer file not found: $tokenizerPath"
                    }

                    val options = Interpreter.Options().apply {
                        numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
                    }
                    val interpreter = Interpreter(modelFile, options)

                    val inputShape = interpreter.getInputTensor(0).shape()
                    val outputShape = interpreter.getOutputTensor(0).shape()
                    val seqLen = inputShape.last()
                    val dimension = outputShape.last()

                    val tokenizer = SpTokenizer(Paths.get(tokenizerPath))
                    val vocabulary = SpVocabulary.from(tokenizer)

                    EmbedderInstance(interpreter, tokenizer, vocabulary, seqLen, dimension, normalize)
                }

                val embedderId = "embedder_${counter.incrementAndGet()}"
                embedders[embedderId] = instance
                result.success(
                    mapOf(
                        "embedderId" to embedderId,
                        "dimension" to instance.dimension,
                        "maxSequenceLength" to instance.seqLen,
                    )
                )
            } catch (e: Exception) {
                result.error(
                    "EMBEDDER_INIT_FAILED",
                    e.message ?: "Embedder initialisation failed",
                    e.javaClass.simpleName
                )
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // embed / embedBatch
    // ──────────────────────────────────────────────────────────────────────

    private fun embed(call: MethodCall, result: Result) {
        val embedderId: String = call.argument("embedderId")
            ?: return result.error("INVALID_ARG", "embedderId is required", null)
        val text: String = call.argument("text")
            ?: return result.error("INVALID_ARG", "text is required", null)
        val promptPrefix: String? = call.argument("promptPrefix")

        val instance = embedders[embedderId]
            ?: return result.error("NOT_FOUND", "No embedder found for id: $embedderId", null)

        scope.launch {
            try {
                val vector = withContext(Dispatchers.IO) {
                    instance.mutex.withLock { runInference(instance, text, promptPrefix) }
                }
                result.success(vector)
            } catch (e: Exception) {
                result.error("EMBEDDING_FAILED", e.message, e.javaClass.simpleName)
            }
        }
    }

    private fun embedBatch(call: MethodCall, result: Result) {
        val embedderId: String = call.argument("embedderId")
            ?: return result.error("INVALID_ARG", "embedderId is required", null)
        val texts: List<String> = call.argument("texts")
            ?: return result.error("INVALID_ARG", "texts is required", null)
        val promptPrefix: String? = call.argument("promptPrefix")

        val instance = embedders[embedderId]
            ?: return result.error("NOT_FOUND", "No embedder found for id: $embedderId", null)

        scope.launch {
            try {
                val vectors = withContext(Dispatchers.IO) {
                    instance.mutex.withLock {
                        texts.map { runInference(instance, it, promptPrefix) }
                    }
                }
                result.success(vectors)
            } catch (e: Exception) {
                result.error("EMBEDDING_FAILED", e.message, e.javaClass.simpleName)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // close
    // ──────────────────────────────────────────────────────────────────────

    private fun close(call: MethodCall, result: Result) {
        val embedderId: String = call.argument("embedderId")
            ?: return result.error("INVALID_ARG", "embedderId is required", null)

        val instance = embedders.remove(embedderId)
            ?: return result.error("NOT_FOUND", "No embedder found for id: $embedderId", null)

        scope.launch {
            try {
                withContext(Dispatchers.IO) { instance.close() }
                result.success(null)
            } catch (e: Exception) {
                result.error("EMBEDDER_CLOSE_FAILED", e.message, null)
            }
        }
    }

    /** Closes all embedders. Called on plugin detach. */
    fun closeAll() {
        embedders.values.forEach { instance ->
            try { instance.close() } catch (_: Exception) {}
        }
        embedders.clear()
    }

    // ──────────────────────────────────────────────────────────────────────
    // Inference
    // ──────────────────────────────────────────────────────────────────────

    // Gemma sentencepiece special token ids.
    private companion object {
        const val PAD_ID = 0
        const val EOS_ID = 1
        const val BOS_ID = 2
    }

    private fun runInference(
        instance: EmbedderInstance,
        text: String,
        promptPrefix: String?,
    ): FloatArray {
        val fullText = (promptPrefix ?: "") + text

        // Tokenize and map pieces to vocabulary ids.
        val pieces = instance.tokenizer.tokenize(fullText)
        val tokenIds = pieces.map { instance.vocabulary.getIndex(it).toInt() }

        // <bos> + tokens + <eos>, truncated then zero-padded to seqLen.
        // (Matches the EmbeddingGemma sentence-transformers tokenizer config.)
        val ids = buildList {
            add(BOS_ID)
            addAll(tokenIds.take(instance.seqLen - 2))
            add(EOS_ID)
        }
        val input = IntArray(instance.seqLen) { i -> if (i < ids.size) ids[i] else PAD_ID }

        val output = ByteBuffer
            .allocateDirect(4 * instance.dimension)
            .order(ByteOrder.nativeOrder())

        if (instance.interpreter.inputTensorCount > 1) {
            // Some exports take (token ids, attention mask) — mask real tokens with 1.
            val mask = IntArray(instance.seqLen) { i -> if (i < ids.size) 1 else 0 }
            instance.interpreter.runForMultipleInputsOutputs(
                arrayOf(arrayOf(input), arrayOf(mask)),
                mapOf(0 to output),
            )
        } else {
            instance.interpreter.run(arrayOf(input), output)
        }

        output.rewind()
        val vector = FloatArray(instance.dimension)
        output.asFloatBuffer().get(vector)

        if (instance.normalize) l2Normalize(vector)
        return vector
    }

    private fun l2Normalize(vector: FloatArray) {
        var sum = 0.0
        for (v in vector) sum += v * v
        val norm = sqrt(sum).toFloat()
        if (norm > 0f) {
            for (i in vector.indices) vector[i] /= norm
        }
    }
}
