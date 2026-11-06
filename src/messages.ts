import type { LanguageModelV3Prompt, LanguageModelV3ToolResultOutput } from "@ai-sdk/provider"

type Text = { type: "text"; text: string }
type Image = { type: "image"; data: string; mimeType: string }
type Part =
  | Text
  | Image
  | { type: "reasoning"; content: Text[] }
  | {
      type: "tool_call"
      id: string
      name: string
      arguments: { type: "json"; raw: string; value: unknown }
    }
type Message = { role: string; content: Part[]; tool_call_id?: string; name?: string; outcome?: string }

export function messages(prompt: LanguageModelV3Prompt): Message[] {
  return prompt.flatMap((message): Message[] => {
    if (message.role === "system")
      return message.content ? [{ role: "system", content: [{ type: "text", text: message.content }] }] : []
    if (message.role === "tool")
      return message.content.flatMap((part): Message[] => {
        if (part.type !== "tool-result") throw new Error("Vibe does not support provider tool approvals")
        return [
          {
            role: "tool",
            tool_call_id: part.toolCallId,
            name: part.toolName,
            outcome:
              part.output.type.startsWith("error-") || part.output.type === "execution-denied" ? "failure" : "success",
            content: output(part.output),
          },
        ]
      })
    const content = message.content.flatMap((part): Part[] => {
      if (part.type === "text") return part.text ? [{ type: "text", text: part.text }] : []
      if (part.type === "reasoning")
        return part.text ? [{ type: "reasoning", content: [{ type: "text", text: part.text }] }] : []
      if (part.type === "tool-call")
        return [
          {
            type: "tool_call",
            id: part.toolCallId,
            name: part.toolName,
            arguments: { type: "json", raw: JSON.stringify(part.input), value: part.input },
          },
        ]
      if (part.type === "file") {
        if (part.data instanceof URL) throw new Error("Vibe requires downloaded image attachments")
        return [
          image(typeof part.data === "string" ? part.data : Buffer.from(part.data).toString("base64"), part.mediaType),
        ]
      }
      throw new Error(`Unsupported Vibe message part: ${part.type}`)
    })
    return content.length ? [{ role: message.role, content }] : []
  })
}

function image(data: string, media: string): Image {
  if (!media.startsWith("image/")) throw new Error("Mistral Vibe supports image attachments, not this file type")
  return { type: "image", data, mimeType: media }
}

function output(value: LanguageModelV3ToolResultOutput): (Text | Image)[] {
  if (value.type === "execution-denied") return [{ type: "text", text: value.reason ?? "Tool execution denied" }]
  if (value.type === "text" || value.type === "error-text") return [{ type: "text", text: value.value }]
  if (value.type === "json" || value.type === "error-json") return [{ type: "text", text: JSON.stringify(value.value) }]
  return value.value.map((part): Text | Image => {
    if (part.type === "text") return { type: "text", text: part.text }
    if (part.type === "image-data" || part.type === "file-data") return image(part.data, part.mediaType)
    throw new Error(`Unsupported Vibe tool output: ${part.type}`)
  })
}
