const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function DELETE(
  _request: Request,
  ctx: RouteContext<"/api/docs/[paper_id]">,
): Promise<Response> {
  const { paper_id } = await ctx.params;
  const upstream = await fetch(
    `${BACKEND_URL}/docs/${encodeURIComponent(paper_id)}`,
    { method: "DELETE" },
  );
  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
