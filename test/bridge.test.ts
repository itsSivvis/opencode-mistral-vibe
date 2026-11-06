import { expect, test } from "bun:test"
import { mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { VibeModel } from "../src/model.js"
import { run } from "../src/process.js"
import type { Event } from "../src/protocol.js"
import { z } from "zod"

// Opt in to the installed Vibe runtime. All HTTP is served locally, with a dummy
// key and isolated VIBE_HOME; no paid requests or user's credentials are involved.
test.skipIf(!process.env.VIBE_TEST_PYTHON)(
  "real Vibe runtime: discovery, streaming, tools, continuation, cancellation and credential isolation",
  async () => {
    const home = await mkdtemp(join(tmpdir(), "opencode-vibe-test-"))
    const keys = ["VIBE_HOME", "OPENCODE_VIBE_PYTHON", "OPENCODE_VIBE_TEST_KEY"] as const
    const saved = Object.fromEntries(keys.map((key) => [key, process.env[key]]))
    const requests: Record<string, unknown>[] = []
    let catalogs = 0
    let offline = false
    const blocked = Promise.withResolvers<void>()
    let mode: "tools" | "answer" | "blocked" | "error" = "tools"
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      idleTimeout: 0,
      async fetch(request) {
        if (new URL(request.url).pathname === "/v1/model-catalog") {
          catalogs++
          expect(request.headers.get("authorization")).toBe(`Bearer ${process.env.OPENCODE_VIBE_TEST_KEY}`)
          expect(new URL(request.url).searchParams.get("surface")).toBe("vibe-code")
          if (offline) return new Response("Unavailable", { status: 503 })
          return Response.json({
            models: [
              {
                id: "fresh-model",
                label: "New hosted model",
                max_context_length: 131072,
                parameters: { reasoning: ["none", "high"] },
                capabilities: { function_calling: true, vision: true },
              },
              {
                id: "chat-only",
                label: "No tools",
                capabilities: { function_calling: false },
              },
            ],
          })
        }
        expect(new URL(request.url).pathname).toBe("/v1/chat/completions")
        expect(request.headers.get("authorization")).toBe("Bearer fixture-only")
        requests.push(z.record(z.string(), z.unknown()).parse(await request.json()))
        if (mode === "error")
          return Response.json({ message: "fixture-only must never appear in errors" }, { status: 401 })
        if (mode === "blocked") {
          blocked.resolve()
          return new Response(
            new ReadableStream({
              start(controller) {
                controller.enqueue(new TextEncoder().encode(": waiting\n\n"))
              },
            }),
            { headers: { "content-type": "text/event-stream" } },
          )
        }
        const chunks =
          mode === "tools"
            ? [
                { delta: { content: "Ich lese 🌍" }, finish_reason: null },
                {
                  delta: {
                    tool_calls: [
                      {
                        index: 0,
                        id: "call12345",
                        type: "function",
                        function: {
                          name: "lookup",
                          arguments: '{"q":"hello"}',
                        },
                      },
                    ],
                  },
                  finish_reason: "tool_calls",
                },
              ]
            : [{ delta: { content: "Fertig" }, finish_reason: "stop" }]
        const body =
          chunks
            .map(
              (choice) =>
                `data: ${JSON.stringify({ id: "fixture", object: "chat.completion.chunk", created: 1, model: "fixture-model", choices: [{ index: 0, ...choice }], usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 } })}\n\n`,
            )
            .join("") + "data: [DONE]\n\n"
        return new Response(body, {
          headers: { "content-type": "text/event-stream" },
        })
      },
    })
    try {
      process.env.VIBE_HOME = home
      process.env.OPENCODE_VIBE_PYTHON = process.env.VIBE_TEST_PYTHON
      process.env.OPENCODE_VIBE_TEST_KEY = "fixture-only"
      await writeFile(
        join(home, "config.toml"),
        `
active_model = "fixture"
[[providers]]
name = "mistral"
backend = "mistral"
api_base = "http://127.0.0.1:${server.port}/v1"
api_key_env_var = "OPENCODE_VIBE_TEST_KEY"
[[models]]
name = "fixture-model"
provider = "mistral"
alias = "fixture"
supports_images = true
max_context_length = 32768
thinking = "off"
`,
      )
      const catalog: Event[] = []
      await run({ type: "probe" }, (event) => catalog.push(event), AbortSignal.timeout(15_000))
      expect(catalog[0]).toMatchObject({
        type: "catalog",
        source: "remote",
        models: expect.arrayContaining([
          expect.objectContaining({
            id: "fixture",
            authenticated: true,
            context: 32768,
          }),
          expect.objectContaining({
            id: "fresh-model",
            authenticated: true,
            context: 131072,
            efforts: ["off", "high"],
          }),
        ]),
      })
      expect(JSON.stringify(catalog)).not.toContain("fixture-only")
      expect(JSON.stringify(catalog)).not.toContain("chat-only")
      const model = new VibeModel("fresh-model")
      const first = await model.doGenerate({
        prompt: [
          { role: "system", content: "Test instructions" },
          { role: "user", content: [{ type: "text", text: "Use lookup" }] },
        ],
        tools: [
          {
            type: "function",
            name: "lookup",
            description: "Look up a value",
            inputSchema: {
              type: "object",
              properties: { q: { type: "string" } },
            },
          },
        ],
      })
      expect(first.finishReason.unified).toBe("tool-calls")
      expect(first.content).toContainEqual({
        type: "text",
        text: "Ich lese 🌍",
      })
      expect(first.content).toContainEqual({
        type: "tool-call",
        toolCallId: "call12345",
        toolName: "lookup",
        input: '{"q":"hello"}',
      })
      expect(requests).toHaveLength(1) // No tool execution or second LLM call inside Vibe.
      expect(requests[0].model).toBe("fresh-model")
      mode = "answer"
      const second = await model.doGenerate({
        prompt: [
          { role: "user", content: [{ type: "text", text: "Use lookup" }] },
          {
            role: "assistant",
            content: [
              {
                type: "tool-call",
                toolCallId: "call12345",
                toolName: "lookup",
                input: { q: "hello" },
              },
            ],
          },
          {
            role: "tool",
            content: [
              {
                type: "tool-result",
                toolCallId: "call12345",
                toolName: "lookup",
                output: { type: "text", value: "world" },
              },
            ],
          },
        ],
      })
      expect(second.content).toEqual([{ type: "text", text: "Fertig" }])
      expect(requests[1].messages).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            role: "tool",
            tool_call_id: "call12345",
            content: "world",
          }),
        ]),
      )
      mode = "error"
      await expect(
        model.doGenerate({
          prompt: [{ role: "user", content: [{ type: "text", text: "Test" }] }],
        }),
      ).rejects.toThrow("bridge failed")
      mode = "blocked"
      const abort = new AbortController()
      const pending = model.doGenerate({
        prompt: [{ role: "user", content: [{ type: "text", text: "Wait" }] }],
        abortSignal: abort.signal,
      })
      const outcome = pending.then(
        () => undefined,
        (error: unknown) => error,
      )
      await blocked.promise
      abort.abort()
      expect(await outcome).toMatchObject({ name: "AbortError" })
      expect(catalogs).toBe(1) // Independent generation processes reuse Vibe's native cache.
      offline = true
      const cached: Event[] = []
      await run({ type: "probe" }, (event) => cached.push(event), AbortSignal.timeout(15_000))
      expect(cached[0]).toMatchObject({
        source: "cache",
        models: expect.arrayContaining([expect.objectContaining({ id: "fresh-model" })]),
      })
      process.env.OPENCODE_VIBE_TEST_KEY = "fixture-other"
      const other: Event[] = []
      await run({ type: "probe" }, (event) => other.push(event), AbortSignal.timeout(15_000))
      expect(other[0]).toMatchObject({ source: "local" })
      expect(JSON.stringify(other)).not.toContain("fresh-model")
    } finally {
      server.stop(true)
      keys.forEach((key) => {
        if (saved[key] === undefined) delete process.env[key]
        else process.env[key] = saved[key]
      })
      await rm(home, { recursive: true, force: true })
    }
  },
)
