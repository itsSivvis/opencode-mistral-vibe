import { expect, test } from "bun:test"
import { VibeModel } from "../src/model.js"
import { offered } from "../src/efforts.js"
import { messages } from "../src/messages.js"
import type { Event } from "../src/protocol.js"

const result: Extract<Event, { type: "result" }> = {
  type: "result",
  parts: [{ type: "text", text: "Hallo 🌍" }],
  finish_reason: "stop",
  usage: { input_tokens: 10, output_tokens: 3, cached_input_tokens: 4, cache_write_tokens: 0, reasoning_tokens: 1 },
}

test("streams text once and reports cache and reasoning usage", async () => {
  const model = new VibeModel("fixture", async (_request, emit) => {
    emit({ type: "reasoning", text: "Think" })
    emit({ type: "text", text: "Hallo " })
    emit({ type: "text", text: "🌍" })
    emit(result)
  })
  const response = await model.doGenerate({ prompt: [{ role: "user", content: [{ type: "text", text: "Hi" }] }] })
  expect(response.content).toEqual([
    { type: "reasoning", text: "Think" },
    { type: "text", text: "Hallo 🌍" },
  ])
  expect(response.usage.inputTokens).toEqual({ total: 10, noCache: 6, cacheRead: 4, cacheWrite: 0 })
  expect(response.usage.outputTokens).toEqual({ total: 3, text: 2, reasoning: 1 })
})

test("preserves roles, tool IDs, errors and image contents", () => {
  const converted = messages([
    { role: "system", content: "Instructions" },
    { role: "user", content: [{ type: "file", mediaType: "image/png", data: new Uint8Array([1, 2, 3]) }] },
    {
      role: "assistant",
      content: [{ type: "tool-call", toolCallId: "call12345", toolName: "read", input: { path: "test" } }],
    },
    {
      role: "tool",
      content: [
        {
          type: "tool-result",
          toolCallId: "call12345",
          toolName: "read",
          output: { type: "error-text", value: "Denied" },
        },
      ],
    },
  ])
  expect(converted.map((item) => item.role)).toEqual(["system", "user", "assistant", "tool"])
  expect(converted[1].content).toEqual([{ type: "image", data: "AQID", mimeType: "image/png" }])
  expect(converted[2].content[0]).toMatchObject({
    type: "tool_call",
    id: "call12345",
    arguments: { raw: '{"path":"test"}', value: { path: "test" } },
  })
  expect(converted[3]).toMatchObject({
    tool_call_id: "call12345",
    outcome: "failure",
    content: [{ type: "text", text: "Denied" }],
  })
})

test("passes tool calls outside the offered catalog through for OpenCode to reject", async () => {
  const model = new VibeModel("fixture", async (request, emit) => {
    expect(request).toMatchObject({ tools: [] })
    emit({
      ...result,
      finish_reason: "tool_call",
      parts: [{ type: "tool_call", id: "call12345", name: "shell", arguments_json: "{}" }],
    })
  })
  const response = await model.doGenerate({
    prompt: [],
    toolChoice: { type: "none" },
    tools: [{ type: "function", name: "shell", inputSchema: {} }],
  })
  expect(response.content).toEqual([{ type: "tool-call", toolCallId: "call12345", toolName: "shell", input: "{}" }])
  expect(response.finishReason).toEqual({ unified: "tool-calls", raw: "tool_call" })
})

test("stream cancellation aborts the owned runner", async () => {
  const stopped = Promise.withResolvers<void>()
  const model = new VibeModel("fixture", async (_request, _emit, signal) => {
    await new Promise<void>((resolve) =>
      signal.addEventListener(
        "abort",
        () => {
          stopped.resolve()
          resolve()
        },
        { once: true },
      ),
    )
  })
  const response = await model.doStream({ prompt: [] })
  await response.stream.cancel()
  await stopped.promise
})

test("truncated responses fail instead of completing successfully", async () => {
  const model = new VibeModel("fixture", async (_request, emit) => {
    emit({ type: "text", text: "Partial" })
  })
  await expect(model.doGenerate({ prompt: [] })).rejects.toThrow("did not return")
})

test("replaces a reasoning level the model refuses and offers the rest", async () => {
  const requests: { effort?: unknown }[] = []
  const run = async (modelId: string, effort: string) => {
    const model = new VibeModel(modelId, async (request, emit) => {
      requests.push(request as { effort?: unknown })
      emit(result)
    })
    await model.doGenerate({
      prompt: [{ role: "user", content: [{ type: "text", text: "Hi" }] }],
      providerOptions: { "mistral-vibe": { effort } },
    })
  }
  await run("deepseek-v4-1-flash-1017", "medium")
  await run("deepseek-v4-1-flash-1017", "low")
  await run("other-model", "medium")
  expect(requests.map((request) => request.effort)).toEqual(["high", "low", "medium"])
  expect(offered("deepseek-v4-1-flash-1017", "medium")).toBe(false)
  expect(offered("deepseek-v4-1-flash-1017", "high")).toBe(true)
  expect(offered("other-model", "medium")).toBe(true)
})
