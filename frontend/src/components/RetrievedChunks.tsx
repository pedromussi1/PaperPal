"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { RetrievedChunk } from "@/lib/api";

type Props = { chunks: RetrievedChunk[] };

/**
 * Devtool view for the chunks the retriever returned. Hidden by default;
 * great talking-point in interviews ("here's exactly what the model saw").
 */
export function RetrievedChunks({ chunks }: Props) {
  const [open, setOpen] = useState(false);

  if (chunks.length === 0) return null;

  return (
    <div className="rounded-md border bg-muted/30">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-sm font-semibold hover:bg-muted/50"
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          Retrieved context
          <Badge variant="secondary" className="text-xs">
            {chunks.length} chunks
          </Badge>
        </span>
        <span className="text-xs text-muted-foreground">
          top score {Math.max(...chunks.map((c) => c.score)).toFixed(3)}
        </span>
      </button>

      {open ? (
        <>
          <Separator />
          <ScrollArea className="h-72">
            <ol className="divide-y">
              {chunks.map((c, i) => (
                <li
                  key={`${c.paper_id}:${c.page}:${c.chunk_idx}`}
                  className="space-y-1 p-3 text-sm"
                >
                  <div className="flex items-center justify-between gap-2 font-mono text-xs text-muted-foreground">
                    <span>
                      #{i + 1} · {c.paper_id.slice(0, 10)}… · page {c.page} · chunk {c.chunk_idx}
                    </span>
                    <span>score {c.score.toFixed(3)}</span>
                  </div>
                  <p className="whitespace-pre-wrap leading-snug">{c.text}</p>
                </li>
              ))}
            </ol>
          </ScrollArea>
        </>
      ) : null}
    </div>
  );
}
