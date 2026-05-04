"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CitedAnswer } from "@/components/Citation";
import { RetrievedChunks } from "@/components/RetrievedChunks";
import { streamQuery, type RetrievedChunk } from "@/lib/api";
import { cn } from "@/lib/utils";

type Message =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      text: string;
      retrieved: RetrievedChunk[];
      status: "streaming" | "done" | "error";
      error?: string;
    };

type Props = { hasDocuments: boolean };

export function Chat({ hasDocuments }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function ask(question: string) {
    setMessages((m) => [
      ...m,
      { role: "user", text: question },
      { role: "assistant", text: "", retrieved: [], status: "streaming" },
    ]);
    setStreaming(true);

    const ctl = new AbortController();
    abortRef.current = ctl;

    try {
      for await (const ev of streamQuery(question, { signal: ctl.signal })) {
        if (ev.type === "retrieved") {
          patchAssistant((m) => ({ ...m, retrieved: ev.chunks }));
        } else if (ev.type === "token") {
          patchAssistant((m) => ({ ...m, text: m.text + ev.text }));
        } else if (ev.type === "done") {
          patchAssistant((m) => ({ ...m, text: ev.answer || m.text, status: "done" }));
        } else if (ev.type === "error") {
          patchAssistant((m) => ({ ...m, status: "error", error: ev.message }));
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      patchAssistant((m) => ({
        ...m,
        status: "error",
        error: ctl.signal.aborted ? "Cancelled." : msg,
      }));
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function patchAssistant(fn: (m: Extract<Message, { role: "assistant" }>) => Message) {
    setMessages((all) => {
      const out = [...all];
      for (let i = out.length - 1; i >= 0; i--) {
        if (out[i].role === "assistant") {
          out[i] = fn(out[i] as Extract<Message, { role: "assistant" }>);
          break;
        }
      }
      return out;
    });
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const q = input.trim();
    if (!q || streaming) return;
    setInput("");
    void ask(q);
  }

  function handleCancel() {
    abortRef.current?.abort();
  }

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1 px-1">
        {messages.length === 0 ? (
          <EmptyState hasDocuments={hasDocuments} />
        ) : (
          <ol className="space-y-6 py-4">
            {messages.map((m, i) => (
              <li key={i}>
                <MessageView message={m} />
              </li>
            ))}
            <div ref={bottomRef} />
          </ol>
        )}
      </ScrollArea>

      <form onSubmit={handleSubmit} className="mt-3 flex items-center gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={
            hasDocuments
              ? "Ask about the uploaded paper(s)…"
              : "Upload a PDF first, then ask a question."
          }
          disabled={streaming || !hasDocuments}
          aria-label="Ask a question"
        />
        {streaming ? (
          <Button type="button" variant="secondary" onClick={handleCancel} aria-label="Cancel">
            <X className="h-4 w-4" />
          </Button>
        ) : (
          <Button type="submit" disabled={!input.trim() || !hasDocuments} aria-label="Send">
            <Send className="h-4 w-4" />
          </Button>
        )}
      </form>
    </div>
  );
}

function MessageView({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg bg-primary px-3.5 py-2.5 text-base font-medium text-primary-foreground">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-lg border bg-card px-3.5 py-2.5 text-base">
        {message.text || message.status === "streaming" ? (
          <CitedAnswer text={message.text} retrieved={message.retrieved} />
        ) : null}
        {message.status === "streaming" && (
          <Loader2 className="ml-1 inline h-3.5 w-3.5 animate-spin text-muted-foreground" />
        )}
        {message.status === "error" && (
          <p className={cn("mt-2 text-sm text-destructive")}>Error: {message.error}</p>
        )}
      </div>
      {message.retrieved.length > 0 && <RetrievedChunks chunks={message.retrieved} />}
    </div>
  );
}

function EmptyState({ hasDocuments }: { hasDocuments: boolean }) {
  return (
    <div className="flex h-full min-h-[24rem] flex-col items-center justify-center gap-2 py-12 text-center text-base text-muted-foreground">
      <p className="text-lg font-semibold text-foreground">PaperPal</p>
      <p className="max-w-md">
        {hasDocuments
          ? "Ask a question about your uploaded paper. Answers will be grounded in retrieved page chunks with inline citations."
          : "Upload a PDF on the left to get started. Born-digital arXiv-style PDFs work best."}
      </p>
    </div>
  );
}
