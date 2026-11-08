import type {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
} from "@ai-sdk/provider"
import { messages } from "./messages.js"
import { run } from "./process.js"
import type { Event } from "./protocol.js"

export const ID = "mistral-vibe"
export const PACKAGE = "@opencode/vibe-agent"

export class VibeModel implements LanguageModelV3 {
  readonly specificationVersion = "v3"
  readonly provider = ID
  readonly supportedUrls = {}

  constructor(
    readonly modelId: string,
    readonly execute: typeof run = run,
  ) {}

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const result = await this.doStream(options)
    const reader = result.stream.getReader()
    const content: LanguageModelV3GenerateResult["content"] = []
    const blocks = new Map<string, { type: "text" | "reasoning"; text: string }>()
    let finish: Extract<LanguageModelV3StreamPart, { type: "finish" }> | undefined
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (value.type === "text-start" || value.type === "reasoning-start") {
        const block = { type: value.type === "text-start" ? ("text" as const) : ("reasoning" as const), text: "" }
        blocks.set(value.id, block)
        content.push(block)
      }
      if (value.type === "text-delta" || value.type === "reasoning-delta") blocks.get(value.id)!.text += value.delta
      if (value.type === "tool-call") content.push(value)
      if (value.type === "finish") finish = value
      if (value.type === "error") throw value.error
    }
    if (!finish) throw new Error("Vibe did not finish its response")
    return { content, finishReason: finish.finishReason, usage: finish.usage, warnings: [] }
  }

  async doStream(options: LanguageModelV3CallOptions) {
    if (options.toolChoice?.type === "required" || options.toolChoice?.type === "tool")
      throw new Error("Vibe does not support forced tool choice")
    if (options.responseFormat?.type === "json") throw new Error("Vibe does not support structured output")
    if (options.tools?.some((tool) => tool.type === "provider"))
      throw new Error("Vibe supports OpenCode function tools only")
    const request = {
      type: "generate",
      model: this.modelId,
      messages: messages(options.prompt),
      tools:
        options.toolChoice?.type === "none"
          ? []
          : (options.tools ?? []).flatMap((tool) =>
              tool.type === "function"
                ? [
                    {
                      name: tool.name,
                      description: tool.description ?? "",
                      parameters: tool.inputSchema,
                    },
                  ]
                : [],
            ),
      maxTokens: options.maxOutputTokens,
      temperature: options.temperature,
      topP: options.topP,
      effort: options.providerOptions?.[ID]?.effort,
    }
    const abort = new AbortController()
    const signal = options.abortSignal ? AbortSignal.any([abort.signal, options.abortSignal]) : abort.signal
    signal.throwIfAborted()
    let cancelled = false
    let task: Promise<void>
    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      start: (controller) => {
        const seen = new Set<"text" | "reasoning">()
        let finish: Extract<Event, { type: "result" }> | undefined
        const delta = (type: "text" | "reasoning", text: string) => {
          if (!text) return
          if (!seen.has(type)) controller.enqueue({ type: `${type}-start`, id: type })
          seen.add(type)
          controller.enqueue({ type: `${type}-delta`, id: type, delta: text })
        }
        controller.enqueue({
          type: "stream-start",
          warnings: ["topK", "seed", "stopSequences", "presencePenalty", "frequencyPenalty"].flatMap((key) =>
            options[key as keyof LanguageModelV3CallOptions] === undefined
              ? []
              : [{ type: "unsupported" as const, feature: key }],
          ),
        })
        task = this.execute(
          request,
          (event) => {
            if (cancelled) return
            if (event.type === "text" || event.type === "reasoning") delta(event.type, event.text)
            if (event.type === "result") finish = event
          },
          signal,
        )
          .then(() => {
            if (cancelled) return
            if (!finish) throw new Error("Vibe did not return a model result")
            const text = finish.parts.flatMap((part) => (part.type === "text" ? [part.text] : [])).join("")
            const reasoning = finish.parts
              .flatMap((part) => (part.type === "reasoning" ? part.content.map((item) => item.text ?? "") : []))
              .join("")
            if (!seen.has("reasoning")) delta("reasoning", reasoning)
            if (!seen.has("text")) delta("text", text)
            seen.forEach((type) => controller.enqueue({ type: `${type}-end`, id: type }))
            for (const part of finish.parts) {
              if (part.type !== "tool_call") continue
              if (!request.tools.some((tool) => tool.name === part.name))
                // Pass the call through: OpenCode answers unoffered tools with a recoverable
                // "No tool named ..." tool error instead of executing them, so the step
                // survives and the model can retry with an offered tool.
                console.warn(`Mistral Vibe requested a tool OpenCode did not offer: ${part.name}`)
              controller.enqueue({
                type: "tool-call",
                toolCallId: part.id,
                toolName: part.name,
                input: part.arguments_json,
              })
            }
            const usage = finish.usage
            controller.enqueue({
              type: "finish",
              finishReason: {
                unified:
                  finish.finish_reason === "tool_call"
                    ? "tool-calls"
                    : finish.finish_reason === "content_filter"
                      ? "content-filter"
                      : finish.finish_reason,
                raw: finish.finish_reason,
              },
              usage: {
                inputTokens: {
                  total: usage.input_tokens,
                  noCache: Math.max(0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens),
                  cacheRead: usage.cached_input_tokens,
                  cacheWrite: usage.cache_write_tokens,
                },
                outputTokens: {
                  total: usage.output_tokens,
                  text: Math.max(0, usage.output_tokens - usage.reasoning_tokens),
                  reasoning: usage.reasoning_tokens,
                },
              },
            })
            controller.close()
          })
          .catch((error) => {
            if (!cancelled) controller.error(error)
          })
      },
      cancel: () => {
        cancelled = true
        abort.abort()
        return task
      },
    })
    return { stream }
  }
}
