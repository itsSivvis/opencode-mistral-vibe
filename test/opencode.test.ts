import { expect, test } from "bun:test"
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"

test.skipIf(!process.env.VIBE_TEST_OPENCODE || !process.env.VIBE_TEST_PYTHON)(
  "installed OpenCode: external connection, model discovery and real Vibe generation",
  async () => {
    const home = await mkdtemp(join(tmpdir(), "opencode-vibe-cli-"))
    const fixture = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      async fetch(request) {
        const body = await request.json()
        expect(body).toMatchObject({ model: "fixture-model", stream: true })
        return new Response(
          `data: ${JSON.stringify({ id: "fixture", object: "chat.completion.chunk", created: 1, model: "fixture-model", choices: [{ index: 0, delta: { content: "VIBE_OK" }, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 } })}\n\ndata: [DONE]\n\n`,
          { headers: { "content-type": "text/event-stream" } },
        )
      },
    })
    await mkdir(join(home, "vibe"))
    await writeFile(
      join(home, "vibe/config.toml"),
      `
active_model = "fixture"
[[providers]]
name = "mistral"
backend = "mistral"
api_base = "http://127.0.0.1:${fixture.port}/v1"
api_key_env_var = "OPENCODE_VIBE_TEST_KEY"
[[models]]
name = "fixture-model"
provider = "mistral"
alias = "fixture"
max_context_length = 32768
thinking = "off"
`,
    )
    const child = Bun.spawn([process.env.VIBE_TEST_OPENCODE!, "serve", "--hostname", "127.0.0.1", "--port", "0"], {
      cwd: home,
      stdin: "ignore",
      stdout: "pipe",
      stderr: "ignore",
      env: {
        PATH: process.env.PATH,
        HOME: home,
        XDG_CONFIG_HOME: join(home, "config"),
        XDG_DATA_HOME: join(home, "data"),
        XDG_CACHE_HOME: join(home, "cache"),
        XDG_STATE_HOME: join(home, "state"),
        OPENCODE_SERVER_PASSWORD: "fixture-server",
        OPENCODE_VIBE_PYTHON: process.env.VIBE_TEST_PYTHON,
        OPENCODE_VIBE_TEST_KEY: "fixture-only",
        VIBE_HOME: join(home, "vibe"),
        OPENCODE_CONFIG_CONTENT: JSON.stringify({
          plugins: [fileURLToPath(new URL("../", import.meta.url))],
          update: "disable",
        }),
      },
    })
    const signal = AbortSignal.timeout(25_000)
    const stop = () => child.kill()
    signal.addEventListener("abort", stop, { once: true })
    try {
      const reader = child.stdout.getReader()
      let output = ""
      let url: string | undefined
      while (!url) {
        const next = await reader.read()
        if (next.done) throw new Error("OpenCode fixture exited before listening")
        output += new TextDecoder().decode(next.value)
        url = /server listening on (http:\/\/[^\s]+)/.exec(output)?.[1]
      }
      reader.releaseLock()
      const headers = {
        authorization: `Basic ${Buffer.from("opencode:fixture-server").toString("base64")}`,
        "content-type": "application/json",
      }
      const integrations = await fetch(`${url}/api/integration`, { headers, signal }).then((response) =>
        response.json(),
      )
      expect(integrations).toMatchObject({
        data: expect.arrayContaining([
          expect.objectContaining({ id: "mistral-vibe", methods: [expect.objectContaining({ type: "external" })] }),
        ]),
      })
      const events = await fetch(`${url}/api/event`, { headers, signal })
      const stream = events.body!.getReader()
      try {
        const connect = await fetch(`${url}/api/integration/mistral-vibe/connect/external`, {
          method: "POST",
          headers,
          signal,
          body: JSON.stringify({ methodID: "vibe-login" }),
        })
        expect(connect.status).toBe(204)
        // The inventory is published asynchronously by the credential event.
        // Wait on that event stream, not elapsed time or a polling interval.
        while (true) {
          const models = await fetch(`${url}/api/model`, { headers, signal }).then((response) => response.text())
          const connection = await fetch(`${url}/api/integration/mistral-vibe`, { headers, signal }).then((response) =>
            response.text(),
          )
          if (connection.includes("needs_auth")) throw new Error(connection)
          if (models.includes('"providerID":"mistral-vibe"') && models.includes('"id":"fixture"')) break
          if ((await stream.read()).done) throw new Error("OpenCode event stream ended before discovery")
        }
      } finally {
        await stream.cancel()
      }
      const generated = await fetch(`${url}/api/experimental/generate/text`, {
        method: "POST",
        headers,
        signal,
        body: JSON.stringify({ prompt: "Say VIBE_OK", model: { providerID: "mistral-vibe", id: "fixture" } }),
      })
      const body = await generated.text()
      expect(generated.status, body).toBe(200)
      expect(body).toContain("VIBE_OK")
    } finally {
      signal.removeEventListener("abort", stop)
      child.kill()
      await child.exited
      fixture.stop(true)
      await rm(home, { recursive: true, force: true })
    }
  },
)
