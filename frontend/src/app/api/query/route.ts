/**
 * SSE proxy: forwards the JSON query payload to FastAPI's /query and pipes
 * the Server-Sent Events stream straight back to the browser. Setting
 * X-Accel-Buffering: no prevents Nginx-style intermediaries from holding
 * the response.
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const body = await request.text();

  const upstream = await fetch(`${BACKEND_URL}/query`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });

  if (!upstream.body) {
    return new Response("Upstream returned no body", { status: 502 });
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
      connection: "keep-alive",
    },
  });
}
