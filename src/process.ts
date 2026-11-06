import { spawn } from "node:child_process"
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

export async function run(request: object, emit: (event: Event) => void, signal: AbortSignal) {
  signal.throwIfAborted()
  const executable = await python()
  signal.throwIfAborted()
  const child = spawn(executable, ["-u", fileURLToPath(new URL("./bridge.py", import.meta.url))], {
    stdio: ["pipe", "pipe", "ignore"],
    windowsHide: true,
  })
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
