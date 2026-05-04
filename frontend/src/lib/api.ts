/**
 * Typed client for the PaperPal backend, accessed through the Next.js
 * route handlers under /api. The route handlers proxy to FastAPI so the
 * browser never needs to know the backend URL or deal with CORS.
 *
 * The /api/query response is a Server-Sent Events stream:
 *   event: retrieved   data: RetrievedChunk[]
 *   event: token       data: { text: string }      (repeated)
 *   event: done        data: { answer: string }
 *   event: error       data: { message: string }
 */

export type RetrievedChunk = {
  paper_id: string;
  page: number;
  chunk_idx: number;
  text: string;
  score: number;
};

export type IngestResponse = {
  paper_id: string;
  title: string | null;
  pages: number;
  chunks: number;
};

export type DocumentSummary = {
  paper_id: string;
  title: string | null;
  pages: number;
  chunks: number;
};

export type DocumentList = { documents: DocumentSummary[] };

export type StreamEvent =
  | { type: "retrieved"; chunks: RetrievedChunk[] }
  | { type: "token"; text: string }
  | { type: "done"; answer: string }
  | { type: "error"; message: string };

export async function uploadPdf(file: File): Promise<IngestResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: fd });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Upload failed (${res.status}): ${detail || res.statusText}`);
  }
  return (await res.json()) as IngestResponse;
}

export async function listDocuments(): Promise<DocumentList> {
  const res = await fetch("/api/docs", { cache: "no-store" });
  if (!res.ok) throw new Error(`docs list failed: ${res.status}`);
  return (await res.json()) as DocumentList;
}

export async function deleteDocument(paperId: string): Promise<void> {
  const res = await fetch(`/api/docs/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 404) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Delete failed (${res.status}): ${detail || res.statusText}`);
  }
}

/**
 * Stream a query response. Yields parsed StreamEvents in order. The
 * stream terminates on `done` or `error`.
 *
 * Cancellation: pass an AbortSignal to interrupt the request.
 */
export async function* streamQuery(
  question: string,
  opts: { paperIds?: string[]; topK?: number; signal?: AbortSignal } = {},
): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      paper_ids: opts.paperIds ?? null,
      top_k: opts.topK ?? null,
    }),
    signal: opts.signal,
  });

  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Query failed (${res.status}): ${detail || res.statusText}`);
  }

  yield* parseSse(res.body);
}

/**
 * Minimal SSE parser. SSE frames are separated by a blank line; each frame
 * has `event:` and `data:` fields. We only care about whole frames, so we
 * buffer until we see `\n\n`.
 */
async function* parseSse(body: ReadableStream<Uint8Array>): AsyncGenerator<StreamEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const ev = parseFrame(frame);
        if (ev) yield ev;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): StreamEvent | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  const raw = dataLines.join("\n");

  try {
    const parsed = JSON.parse(raw);
    switch (event) {
      case "retrieved":
        return { type: "retrieved", chunks: parsed as RetrievedChunk[] };
      case "token":
        return { type: "token", text: (parsed as { text: string }).text };
      case "done":
        return { type: "done", answer: (parsed as { answer: string }).answer };
      case "error":
        return { type: "error", message: (parsed as { message: string }).message };
      default:
        return null;
    }
  } catch {
    return { type: "error", message: `Malformed SSE frame: ${raw.slice(0, 200)}` };
  }
}
