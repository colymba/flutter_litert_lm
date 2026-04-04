package dev.colymba.flutter_litert_lm

import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.tool
import io.flutter.plugin.common.BinaryMessenger
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger

/**
 * ConversationHandler
 *
 * Handles all `conversation/...` MethodChannel calls.
 *
 * Conversations are identified by a string ID returned to Dart. Streaming
 * responses are delivered via a dynamically-registered [EventChannel] whose
 * name is also returned to Dart.
 */
class ConversationHandler(
    private val scope: CoroutineScope,
    private val messenger: BinaryMessenger,
) {
    data class ConversationEntry(
        val conversation: Conversation,
        val engine: Engine,
    )

    private val conversations = ConcurrentHashMap<String, ConversationEntry>()
    private val streamChannels = ConcurrentHashMap<String, EventChannel>()
    private val counter = AtomicInteger(0)

    // We keep a reference to EngineHandler for engine lookups.
    // Set by FlutterLiteRtLmPlugin after construction.
    var engineHandler: EngineHandler? = null

    // ──────────────────────────────────────────────────────────────────────
    // DartOpenApiTool — bridges a Dart tool declaration to the SDK
    // ──────────────────────────────────────────────────────────────────────

    /**
     * An [OpenApiTool] whose JSON schema comes from Dart. The `execute` method
     * is never called because `automaticToolCalling` is always false when tools
     * are registered via Dart.
     */
    private class DartOpenApiTool(
        private val name: String,
        private val description: String,
        private val parametersJson: String,
    ) : OpenApiTool {
        override fun getToolDescriptionJsonString(): String =
            """{"name":"$name","description":"$description","parameters":$parametersJson}"""

        override fun execute(paramsJsonString: String): String {
            // Should never be called: automaticToolCalling is false.
            error("DartOpenApiTool.execute() should not be called — tool execution is handled in Dart.")
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Dispatch
    // ──────────────────────────────────────────────────────────────────────

    fun handle(call: MethodCall, result: Result) {
        when (call.method) {
            "conversation/create"            -> create(call, result)
            "conversation/sendMessage"       -> sendMessage(call, result)
            "conversation/sendMessageStream" -> sendMessageStream(call, result)
            "conversation/cancel"            -> cancel(call, result)
            "conversation/close"             -> close(call, result)
            else                             -> result.notImplemented()
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // create
    // ──────────────────────────────────────────────────────────────────────

    @Suppress("UNCHECKED_CAST")
    private fun create(call: MethodCall, result: Result) {
        val engineId: String = call.argument("engineId")
            ?: return result.error("INVALID_ARG", "engineId is required", null)

        val engine = engineHandler?.getEngine(engineId)
            ?: return result.error("NOT_FOUND", "No engine found for id: $engineId", null)

        val systemInstruction: String? = call.argument("systemInstruction")
        val topK: Int? = call.argument("topK")
        val topP: Double? = call.argument("topP")
        val temperature: Double? = call.argument("temperature")
        val maxOutputTokens: Int? = call.argument("maxOutputTokens")
        val automaticToolCalling: Boolean = call.argument("automaticToolCalling") ?: true
        val toolsArg: List<Map<String, Any?>>? = call.argument("tools")

        scope.launch {
            try {
                val conversation = withContext(Dispatchers.IO) {
                    val samplerConfig = SamplerConfig(
                        topK = topK ?: 40,
                        topP = topP ?: 0.95,
                        temperature = temperature ?: 0.8,
                        seed = 0
                    )

                    val toolInstances = toolsArg?.map { toolMap ->
                        val tName = toolMap["name"] as String
                        val tDesc = toolMap["description"] as String
                        val tParamsJson = toolMap["parametersJson"] as String
                        tool(DartOpenApiTool(tName, tDesc, tParamsJson))
                    } ?: emptyList()

                    val config = ConversationConfig(
                        systemInstruction = systemInstruction?.let { Contents.of(it) },
                        samplerConfig = samplerConfig,
                        tools = toolInstances,
                        automaticToolCalling = automaticToolCalling,
                    )

                    engine.createConversation(config)
                }

                val conversationId = "conversation_${counter.incrementAndGet()}"
                conversations[conversationId] = ConversationEntry(conversation, engine)
                result.success(conversationId)

            } catch (e: Exception) {
                result.error(
                    "CONVERSATION_CREATE_FAILED",
                    e.message ?: "Conversation creation failed",
                    e.javaClass.simpleName
                )
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // sendMessage (blocking)
    // ──────────────────────────────────────────────────────────────────────

    private fun sendMessage(call: MethodCall, result: Result) {
        val conversationId: String = call.argument("conversationId")
            ?: return result.error("INVALID_ARG", "conversationId is required", null)

        val partsArg: List<Map<String, Any?>>? = call.argument("parts")
            ?: return result.error("INVALID_ARG", "parts is required", null)

        val role: String? = call.argument("role")

        val entry = conversations[conversationId]
            ?: return result.error("NOT_FOUND", "No conversation for id: $conversationId", null)

        scope.launch {
            try {
                val responseMessage = withContext(Dispatchers.IO) {
                    val contents = buildContents(partsArg!!)
                    if (role == "tool") {
                        // Tool-response round-trip: send a Message.tool()
                        entry.conversation.sendMessage(Message.tool(contents))
                    } else {
                        entry.conversation.sendMessage(contents)
                    }
                }
                result.success(messageToMap(responseMessage))
            } catch (e: Exception) {
                result.error("SEND_MESSAGE_FAILED", e.message, null)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // sendMessageStream (async via EventChannel)
    // ──────────────────────────────────────────────────────────────────────

    private fun sendMessageStream(call: MethodCall, result: Result) {
        val conversationId: String = call.argument("conversationId")
            ?: return result.error("INVALID_ARG", "conversationId is required", null)

        val partsArg: List<Map<String, Any?>>? = call.argument("parts")
            ?: return result.error("INVALID_ARG", "parts is required", null)

        val entry = conversations[conversationId]
            ?: return result.error("NOT_FOUND", "No conversation for id: $conversationId", null)

        // Register a dynamic EventChannel for this stream.
        val streamChannelName = "flutter_litert_lm/stream/$conversationId"
        val eventChannel = EventChannel(messenger, streamChannelName)
        streamChannels[conversationId] = eventChannel

        eventChannel.setStreamHandler(object : EventChannel.StreamHandler {
            override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
                // Start async inference on IO thread when Dart subscribes.
                scope.launch {
                    try {
                        val contents = buildContents(partsArg!!)
                        withContext(Dispatchers.IO) {
                            entry.conversation.sendMessageAsync(
                                contents,
                                object : MessageCallback {
                                    private var lastMessage: Message? = null
                                    
                                    override fun onMessage(message: Message) {
                                        lastMessage = message
                                        val chunk = message.contents.contents
                                            .filterIsInstance<Content.Text>()
                                            .joinToString("") { it.text }
                                        scope.launch(Dispatchers.Main) {
                                            events.success(chunk)
                                        }
                                    }

                                    override fun onDone() {
                                        scope.launch(Dispatchers.Main) {
                                            // Emit terminal tool-call event if present in the last received message.
                                            val message = lastMessage
                                            if (message != null && message.toolCalls.isNotEmpty()) {
                                                val calls = message.toolCalls.map { tc ->
                                                    mapOf(
                                                        "name" to tc.name,
                                                        "argumentsJson" to tc.arguments
                                                    )
                                                }
                                                events.success(
                                                    mapOf("_type" to "toolCalls", "calls" to calls)
                                                )
                                            }
                                            events.endOfStream()
                                        }
                                    }

                                    override fun onError(error: Throwable) {
                                        scope.launch(Dispatchers.Main) {
                                            events.error(
                                                "STREAM_ERROR",
                                                error.message ?: "Streaming error",
                                                null
                                            )
                                        }
                                    }
                                }
                            )
                        }
                    } catch (e: Exception) {
                        scope.launch(Dispatchers.Main) {
                            events.error("STREAM_START_FAILED", e.message, null)
                        }
                    }
                }
            }

            override fun onCancel(arguments: Any?) {
                cleanupStreamChannel(conversationId)
            }
        })

        // Return the channel name to Dart so it can subscribe.
        result.success(streamChannelName)
    }

    // ──────────────────────────────────────────────────────────────────────
    // cancel
    // ──────────────────────────────────────────────────────────────────────

    private fun cancel(call: MethodCall, result: Result) {
        val conversationId: String = call.argument("conversationId")
            ?: return result.error("INVALID_ARG", "conversationId is required", null)

        val entry = conversations[conversationId] ?: return result.success(null)

        scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    entry.conversation.cancelProcess()
                }
                result.success(null)
            } catch (e: Exception) {
                result.error("CANCEL_FAILED", e.message, null)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // close
    // ──────────────────────────────────────────────────────────────────────

    private fun close(call: MethodCall, result: Result) {
        val conversationId: String = call.argument("conversationId")
            ?: return result.error("INVALID_ARG", "conversationId is required", null)

        val entry = conversations.remove(conversationId)
            ?: return result.success(null) // already closed — idempotent

        cleanupStreamChannel(conversationId)

        scope.launch {
            try {
                withContext(Dispatchers.IO) { entry.conversation.close() }
                result.success(null)
            } catch (e: Exception) {
                result.error("CONVERSATION_CLOSE_FAILED", e.message, null)
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────

    /** Builds a [Contents] object from the Dart-serialised parts list. */
    @Suppress("UNCHECKED_CAST")
    private fun buildContents(parts: List<Map<String, Any?>>): Contents {
        val contentList = parts.map { part ->
            when (val type = part["type"] as? String) {
                "text"         -> Content.Text(part["text"] as String)
                "imageFile"    -> Content.ImageFile(part["path"] as String)
                "imageBytes"   -> Content.ImageBytes(part["data"] as ByteArray)
                "audioFile"    -> Content.AudioFile(part["path"] as String)
                "audioBytes"   -> Content.AudioBytes(part["data"] as ByteArray)
                "toolResponse" -> Content.ToolResponse(
                    part["toolName"] as String,
                    part["responseJson"] as String,
                )
                else           -> throw IllegalArgumentException("Unknown content type: $type")
            }
        }
        return Contents.of(*contentList.toTypedArray())
    }

    /**
     * Converts a native [Message] to a map suitable for returning over the
     * MethodChannel. Includes both text and tool-call information.
     */
    private fun messageToMap(message: Message): Map<String, Any> {
        val text = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString("") { it.text }
        val toolCalls = message.toolCalls.map { tc ->
            mapOf("name" to tc.name, "argumentsJson" to tc.arguments)
        }
        return mapOf("text" to text, "toolCalls" to toolCalls)
    }

    private fun cleanupStreamChannel(conversationId: String) {
        streamChannels.remove(conversationId)?.setStreamHandler(null)
    }

    /** Closes all conversations. Called on plugin detach. */
    fun closeAll() {
        conversations.values.forEach { entry ->
            try { entry.conversation.close() } catch (_: Exception) {}
        }
        conversations.clear()
        streamChannels.values.forEach { ch -> ch.setStreamHandler(null) }
        streamChannels.clear()
    }
}
