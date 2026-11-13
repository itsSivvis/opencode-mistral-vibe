// Mistral's endpoints accept fewer reasoning levels than Vibe's generic model
// config lists. Selecting an unsupported level makes Mistral answer HTTP 400
// ("Mistral Vibe bridge failed (SDKError)"). Verified live: medium is refused by
// deepseek-v4-1-flash-1017, which accepts none, low, high, xhigh and max.
//
// Each entry maps a refused level to the closest accepted one.
const FALLBACKS: Record<string, Record<string, string>> = {
  "deepseek-v4-1-flash-1017": { medium: "high" },
}

/** Whether the level should be offered to the user for this model. */
export function offered(model: string, effort: string): boolean {
  return !(effort in (FALLBACKS[model] ?? {}))
}

/**
 * The level to send to Mistral. A level the model refuses can still arrive from a
 * session or preference saved before it was hidden, so replace it instead of failing.
 */
export function effective(model: string, effort: string | undefined): string | undefined {
  return effort === undefined ? undefined : (FALLBACKS[model]?.[effort] ?? effort)
}
