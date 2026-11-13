import { spawn, type ChildProcessByStdio } from "node:child_process"
import type { Readable, Writable } from "node:stream"
import { access, readFile, realpath } from "node:fs/promises"
import { constants } from "node:fs"
import { delimiter, dirname, join } from "node:path"
import { homedir } from "node:os"
import { fileURLToPath } from "node:url"
import { StringDecoder } from "node:string_decoder"
import { Event } from "./protocol.js"

export async function python() {
  if (process.env.OPENCODE_VIBE_PYTHON) return process.env.OPENCODE_VIBE_PYTHON
  const paths = process.env.OPENCODE_VIBE_PATH
    ? [process.env.OPENCODE_VIBE_PATH]
    : [
        ...(process.env.PATH ?? "").split(delimiter).map((path) => join(path, "vibe")),
        join(homedir(), ".local/bin/vibe"),
      ]
  for (const path of paths) {
    if (
      !(await access(path, constants.X_OK).then(
        () => true,
        () => false,
      ))
    )
      continue
    const target = await realpath(path)
    const sibling = join(dirname(target), process.platform === "win32" ? "python.exe" : "python")
    if (
      await access(sibling, constants.X_OK).then(
        () => true,
        () => false,
      )
    )
      return sibling
    const line = (await readFile(target, "utf8")).split("\n", 1)[0]
    if (line?.startsWith("#!/") && !line.includes(" ")) return line.slice(2)
  }
  throw new Error(
    "Mistral Vibe's Python environment was not found. Install with uv tool install mistral-vibe==3.1.0, or set OPENCODE_VIBE_PYTHON.",
  )
}

const BRIDGE = fileURLToPath(new URL("./bridge.py", import.meta.url))
// Set OPENCODE_VIBE_PREWARM=0 to go back to one cold Python process per request.
const PREWARM = process.env.OPENCODE_VIBE_PREWARM !== "0"
const WARM_IDLE_MS = 10 * 60 * 1000

// One idle worker with Vibe already imported. A request takes it, runs once and
// exits exactly like a cold process; a fresh one is started for the next request.
type Worker = ChildProcessByStdio<Writable, Readable, null>
let warm: { child: Worker; executable: string; env: string; timer: ReturnType<typeof setTimeout> } | undefined

// A warm worker keeps the environment it was started with, so it is only reused while that is unchanged.
const environment = () => JSON.stringify(process.env)

function discard() {
  if (!warm) return
  clearTimeout(warm.timer)
  warm.child.kill("SIGKILL")
  warm = undefined
}

function prewarm(executable: string) {
  if (!PREWARM || warm) return
  try {
    const child = spawn(executable, ["-u", BRIDGE, "--warm"], { stdio: ["pipe", "pipe", "ignore"], windowsHide: true })
    const entry = { child, executable, env: environment(), timer: setTimeout(discard, WARM_IDLE_MS) }
    entry.timer.unref?.()
    child.unref()
    child.on("error", () => {})
    child.stdin.on("error", () => {})
    child.stdout.on("data", () => {}) // Nothing is expected before a request; drop it.
    child.once("close", () => {
      clearTimeout(entry.timer)
      if (warm === entry) warm = undefined
    })
    warm = entry
  } catch {
    warm = undefined
  }
}

// Only model requests are worth a standby worker; a catalog probe alone is not.
function acquire(executable: string, standby: boolean): Worker {
  if (
    PREWARM &&
    warm &&
    warm.executable === executable &&
    warm.env === environment() &&
    warm.child.exitCode === null &&
    !warm.child.killed
  ) {
    const { child, timer } = warm
    clearTimeout(timer)
    warm = undefined
    child.ref()
    child.stdout.removeAllListeners("data")
    if (standby) prewarm(executable)
    return child
  }
  discard()
  const child = spawn(executable, ["-u", BRIDGE], { stdio: ["pipe", "pipe", "ignore"], windowsHide: true })
  if (standby) prewarm(executable)
  return child
}

process.on("exit", discard)

/** Stop the standby worker, e.g. when the plugin is disposed or reloaded. */
export const shutdown = discard

export async function run(request: object, emit: (event: Event) => void, signal: AbortSignal) {
  signal.throwIfAborted()
  const executable = await python()
  signal.throwIfAborted()
  const child = acquire(executable, (request as { type?: unknown }).type === "generate")
  let buffer = ""
  const decoder = new StringDecoder("utf8")
  let terminal = false
  let failure: Error | undefined
  const stop = () => child.kill("SIGKILL")
  const complete = new Promise<void>((resolve, reject) => {
    child.on("error", () => {
      failure = new Error("Cannot start the Mistral Vibe Python environment. Check OPENCODE_VIBE_PYTHON.")
    })
    child.stdin.on("error", () => {}) // Process exit/abort is reported by close.
    child.stdout.on("data", (data: Buffer) => {
      buffer += decoder.write(data)
      if (buffer.length > 32 * 1024 * 1024) {
        failure = new Error("Mistral Vibe returned an oversized protocol message")
        stop()
        return
      }
      const lines = buffer.split("\n")
      buffer = lines.pop()!
      for (const line of lines) {
        if (!line.trim() || failure) continue
        try {
          const event = Event.parse(JSON.parse(line))
          if (terminal) throw new Error("Data after final Vibe response")
          if (event.type === "error") throw new Error(event.message)
          terminal = event.type === "catalog" || event.type === "result"
          emit(event)
        } catch (error) {
          // Schema errors may include response values; do not expose them.
          failure =
            error instanceof Error && error.name === "Error"
              ? error
              : new Error(
                  `Invalid response from the Mistral Vibe bridge (${error instanceof Error ? error.name : "unknown"})`,
                )
          stop()
        }
      }
    })
    child.once("close", (code) => {
      signal.removeEventListener("abort", stop)
      if (signal.aborted) return reject(signal.reason)
      if (failure) return reject(failure)
      if (code !== 0 || !terminal || buffer.trim())
        return reject(new Error("Mistral Vibe stopped before completing its response"))
      resolve()
    })
  })
  signal.addEventListener("abort", stop, { once: true })
  if (signal.aborted) stop()
  child.stdin.end(JSON.stringify(request) + "\n")
  return complete
}
