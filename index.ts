import { Model, Plugin, Provider } from "@opencode/plugin"
import { ID, PACKAGE, VibeModel } from "./src/model.js"
import { Catalog } from "./src/protocol.js"
import { run } from "./src/process.js"

export default Plugin.define({
  id: "opencode.provider.mistral-vibe",
  async setup(ctx) {
    if (ctx.location.workspaceID !== undefined) return
    const id = Provider.ID.make(ID)
    const saved = Catalog.safeParse(await ctx.storage.get("catalog"))
    let catalog: Catalog = saved.success ? saved.data : { type: "catalog", version: "", models: [] }
    await ctx.provider.transform((editor) => {
      editor.add({
        info: { ...Provider.Info.empty(id), name: "Mistral Vibe", package: `aisdk:${PACKAGE}` },
        models: catalog.models.map((model) => ({
          ...Model.Info.default(id, Model.ID.make(model.id)),
          name: model.name,
          limit: { context: model.context, output: model.output },
          capabilities: { tools: true, input: model.images ? ["text", "image"] : ["text"], output: ["text"] },
          variants: model.efforts.map((effort) => ({ id: Model.VariantID.make(effort), settings: { effort } })),
          // A Vibe key may be subscription-backed. Do not invent API charges.
          cost: [],
        })),
      })
    })
    await ctx.integration.transform((editor) => {
      editor.update(ID, (integration) => {
        integration.name = "Mistral Vibe"
      })
      editor.method.update({
        integrationID: ID,
        method: { id: "vibe-login", type: "external", label: "Use Vibe login (vibe --setup)" },
      })
    })
    await ctx.aisdk.hook(
      "sdk",
      (event) => {
        if (event.package !== PACKAGE) return
        event.sdk = { languageModel: (model: string) => new VibeModel(model) }
      },
      { providerID: ID },
    )
    const abort = new AbortController()
    const refresh = async () => {
      const connection = await ctx.integration.connection.active(ID)
      if (!connection) return
      const problem = await run(
        { type: "probe" },
        (event) => {
          if (event.type === "catalog") catalog = event
        },
        AbortSignal.any([abort.signal, AbortSignal.timeout(30_000)]),
      ).then(
        () =>
          catalog.models.some((model) => model.authenticated)
            ? undefined
            : "Mistral Vibe is not logged in. Run vibe --setup on the OpenCode server, then reconnect Mistral Vibe.",
        (error: unknown) => (error instanceof Error ? error.message : "Mistral Vibe could not be checked"),
      )
      if (abort.signal.aborted) return
      await ctx.storage.set("catalog", catalog)
      await ctx.provider.reload()
      await ctx.integration.connection.status({
        integrationID: ID,
        connection,
        status: problem ? { status: "needs_auth", message: problem } : undefined,
      })
    }
    const events = (async () => {
      for await (const event of ctx.event.subscribe({ types: ["credential.switched"], signal: abort.signal })) {
        if (event.type === "credential.switched" && event.data.integrationID === ID) await refresh()
      }
    })().catch((error: unknown) => {
      if (!abort.signal.aborted)
        console.error("Mistral Vibe refresh failed", error instanceof Error ? error.name : "Error")
    })
    await refresh()
    return async () => {
      abort.abort()
      await events
    }
  },
})
