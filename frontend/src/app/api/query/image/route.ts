/**
 * Multipart proxy: forwards the chat-image FormData to FastAPI's
 * /query/image and pipes the SSE response straight back to the browser.
 *
 * We re-build the FormData rather than streaming the request body because
 * Next.js route handlers don't yet have a stable streaming-body forwarding
 * API in v16, and an image upload is small enough that the in-memory round
 * trip is fine.
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const incoming = await request.formData();

  const outgoing = new FormData();
  for (const [key, value] of incoming.entries()) {
    outgoing.append(key, value);
  }

  const upstream = await fetch(`${BACKEND_URL}/query/image`, {
    method: "POST",
    body: outgoing,
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
