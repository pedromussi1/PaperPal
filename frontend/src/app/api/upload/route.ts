/**
 * Multipart proxy: forwards the uploaded file to the FastAPI backend's
 * /upload endpoint. Keeps the browser on the same origin so we don't deal
 * with CORS, and hides the backend URL from the client bundle.
 */

const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request): Promise<Response> {
  const formData = await request.formData();

  const upstream = await fetch(`${BACKEND_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json" },
  });
}
