"use client";

import { useCallback, useRef, useState } from "react";
import { FileUp, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uploadPdf, type IngestResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

type Props = {
  onUploaded: (result: IngestResponse) => void;
};

export function Upload({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(
    async (file: File) => {
      setError(null);
      setBusy(true);
      try {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          throw new Error("Only .pdf files are supported.");
        }
        const result = await uploadPdf(file);
        onUploaded(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(false);
      }
    },
    [onUploaded],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) void submit(file);
      }}
      className={cn(
        "rounded-lg border-2 border-dashed p-6 text-center transition-colors",
        dragOver
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/25 hover:border-muted-foreground/50",
        busy && "pointer-events-none opacity-60",
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) void submit(file);
          e.target.value = "";
        }}
      />

      <div className="flex flex-col items-center gap-3">
        {busy ? (
          <>
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            <p className="text-base text-muted-foreground">
              Ingesting PDF — this may take a moment.
            </p>
          </>
        ) : (
          <>
            <FileUp className="h-8 w-8 text-muted-foreground" />
            <div className="space-y-1">
              <p className="text-base font-semibold">Drop a PDF here</p>
              <p className="text-sm text-muted-foreground">
                Born-digital arXiv-style papers ingest cleanly. Scanned PDFs are not supported in v1.
              </p>
            </div>
            <Button variant="secondary" size="sm" onClick={() => inputRef.current?.click()}>
              Choose file
            </Button>
          </>
        )}
        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  );
}
