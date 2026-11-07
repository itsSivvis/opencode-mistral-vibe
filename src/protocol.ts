import { z } from "zod"

export const Catalog = z.object({
  type: z.literal("catalog"),
  version: z.string(),
  source: z.enum(["remote", "cache", "local"]).optional(),
  models: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      context: z.number().nonnegative(),
      output: z.number().nonnegative(),
      images: z.boolean(),
      efforts: z.array(z.string()),
      authenticated: z.boolean(),
      hidden: z.boolean().optional(),
    }),
  ),
})
export type Catalog = z.infer<typeof Catalog>

export const Event = z.discriminatedUnion("type", [
  Catalog,
  z.object({ type: z.literal("text"), text: z.string() }),
  z.object({ type: z.literal("reasoning"), text: z.string() }),
  z.object({
    type: z.literal("error"),
    message: z.string(),
    status: z.number().optional(),
  }),
  z.object({
    type: z.literal("result"),
    parts: z.array(
      z.discriminatedUnion("type", [
        z.object({ type: z.literal("text"), text: z.string() }),
        z.object({
          type: z.literal("reasoning"),
          content: z.array(z.object({ type: z.string(), text: z.string().optional() })),
        }),
        z.object({
          type: z.literal("tool_call"),
          id: z.string(),
          name: z.string(),
          arguments_json: z.string(),
        }),
      ]),
    ),
    finish_reason: z.enum(["stop", "tool_call", "length", "content_filter", "other"]),
    usage: z.object({
      input_tokens: z.number(),
      output_tokens: z.number(),
      cached_input_tokens: z.number(),
      cache_write_tokens: z.number(),
      reasoning_tokens: z.number(),
    }),
  }),
])
export type Event = z.infer<typeof Event>
