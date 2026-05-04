"use client";

import { useEffect, useState } from "react";
import { FileText, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Chat } from "@/components/Chat";
import { Upload } from "@/components/Upload";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  deleteDocument,
  listDocuments,
  type DocumentSummary,
  type IngestResponse,
} from "@/lib/api";

export default function HomePage() {
  const [docs, setDocs] = useState<DocumentSummary[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    listDocuments()
      .then((d) => {
        if (alive) setDocs(d.documents);
      })
      .catch(() => {
        // Backend may not be running yet; the empty state handles it gracefully.
      });
    return () => {
      alive = false;
    };
  }, []);

  function handleUploaded(result: IngestResponse) {
    setDocs((cur) => {
      const filtered = cur.filter((d) => d.paper_id !== result.paper_id);
      return [
        ...filtered,
        {
          paper_id: result.paper_id,
          title: result.title,
          pages: result.pages,
          chunks: result.chunks,
        },
      ];
    });
  }

  async function handleDelete(paperId: string) {
    setDeleting(paperId);
    try {
      await deleteDocument(paperId);
      setDocs((cur) => cur.filter((d) => d.paper_id !== paperId));
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <main className="mx-auto flex h-[100dvh] w-full max-w-6xl flex-col gap-4 px-4 py-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">PaperPal</h1>
          <p className="text-sm text-muted-foreground">
            Local RAG over your research papers · Llama 3.1 8B + ChromaDB
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-mono text-xs">
            100% local
          </Badge>
          <ThemeToggle />
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[20rem_1fr]">
        <aside className="flex flex-col gap-4">
          <Upload onUploaded={handleUploaded} />
          <Card className="flex min-h-0 flex-1 flex-col p-3">
            <h2 className="mb-2 text-sm font-semibold text-muted-foreground">
              Library ({docs.length})
            </h2>
            {docs.length === 0 ? (
              <p className="text-sm italic text-muted-foreground">
                No papers ingested yet.
              </p>
            ) : (
              <ul className="space-y-2 overflow-y-auto">
                {docs.map((d) => (
                  <li
                    key={d.paper_id}
                    className="group flex items-start gap-2 rounded-md border bg-muted/30 p-2 text-sm"
                  >
                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-semibold" title={d.title ?? undefined}>
                        {d.title ?? "Untitled"}
                      </p>
                      <p className="font-mono text-xs text-muted-foreground">
                        {d.paper_id.slice(0, 10)}… · {d.pages} pages · {d.chunks} chunks
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100"
                      aria-label={`Delete ${d.title ?? d.paper_id}`}
                      disabled={deleting === d.paper_id}
                      onClick={() => {
                        if (
                          confirm(
                            `Delete "${d.title ?? d.paper_id}"? This removes it from the index.`,
                          )
                        ) {
                          void handleDelete(d.paper_id);
                        }
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </aside>

        <Card className="flex min-h-0 flex-col p-4">
          <Chat hasDocuments={docs.length > 0} />
        </Card>
      </div>
    </main>
  );
}
