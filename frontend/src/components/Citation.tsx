"use client";

import { Fragment, type ReactNode } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { RetrievedChunk } from "@/lib/api";

type Props = {
  text: string;
  retrieved: RetrievedChunk[];
};

const CITATION_RE = /\[([0-9a-f]{6,32}):(\d+)\]/g;

/**
 * Render an answer with inline citations parsed from `[paper_id:page]`
 * markers. Each marker becomes a clickable superscript number; clicking
 * pops a snippet of the matching retrieved chunk(s).
 *
 * The numbering is per-render: the first distinct paper:page seen becomes
 * [1], the next [2], etc. Repeated references re-use the same number.
 */
export function CitedAnswer({ text, retrieved }: Props) {
  const refMap = new Map<string, number>();

  const findChunks = (paperId: string, page: number) =>
    retrieved.filter((c) => c.paper_id === paperId && c.page === page);

  const out: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  CITATION_RE.lastIndex = 0;

  while ((match = CITATION_RE.exec(text)) !== null) {
    const [marker, paperId, pageStr] = match;
    const page = Number(pageStr);
    const key = `${paperId}:${page}`;

    if (cursor < match.index) {
      out.push(<Fragment key={`t-${cursor}`}>{text.slice(cursor, match.index)}</Fragment>);
    }

    if (!refMap.has(key)) refMap.set(key, refMap.size + 1);
    const refNum = refMap.get(key)!;
    const chunks = findChunks(paperId, page);

    out.push(
      <Popover key={`c-${match.index}`}>
        <PopoverTrigger
          className="mx-0.5 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded bg-primary/15 px-1 text-xs font-semibold leading-none text-primary align-super hover:bg-primary/25 focus:outline-none focus:ring-2 focus:ring-primary/40"
          aria-label={`Citation for page ${page}`}
        >
          {refNum}
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="w-96 max-w-[90vw] text-sm"
        >
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="font-semibold">Page {page}</span>
            <span className="font-mono text-xs text-muted-foreground">
              {paperId.slice(0, 10)}…
            </span>
          </div>
          {chunks.length === 0 ? (
            <p className="italic text-muted-foreground">
              No retrieved chunk matches this citation. The model may have
              hallucinated this reference.
            </p>
          ) : (
            <ul className="space-y-3">
              {chunks.map((c) => (
                <li
                  key={`${c.paper_id}:${c.page}:${c.chunk_idx}`}
                  className="border-l-2 border-primary/40 pl-2"
                >
                  <p className="whitespace-pre-wrap leading-snug text-foreground/90">
                    {c.text.slice(0, 600)}
                    {c.text.length > 600 ? "…" : ""}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    score {c.score.toFixed(3)} · chunk {c.chunk_idx}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </PopoverContent>
      </Popover>,
    );
    out.push(
      <span key={`m-${match.index}`} className="sr-only">
        {marker}
      </span>,
    );

    cursor = match.index + marker.length;
  }

  if (cursor < text.length) {
    out.push(<Fragment key={`t-${cursor}-tail`}>{text.slice(cursor)}</Fragment>);
  }

  return <span className="whitespace-pre-wrap leading-relaxed">{out}</span>;
}
