"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Paperclip, Send, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { CitedAnswer } from "@/components/Citation";
import { RetrievedChunks } from "@/components/RetrievedChunks";
import {
  streamImageQuery,
  streamQuery,
  type RetrievedChunk,
  type StreamEvent,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type UserMessage = {
  role: "user";
  text: string;
  image?: {
    previewUrl: string; // data URL — survives localStorage round-trips
    ocrText?: string; // populated once the backend emits the `ocr` event
  };
};

type AssistantMessage = {
  role: "assistant";
  text: string;
  retrieved: RetrievedChunk[];
  status: "streaming" | "done" | "error";
  error?: string;
};

type Message = UserMessage | AssistantMessage;

type Props = { hasDocuments: boolean };

const STORAGE_KEY = "paperpal:chat:v2"; // bumped: schema now includes user.image
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

export function Chat({ hasDocuments }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollRootRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);

  // Hydrate from localStorage on mount. Any message left in 'streaming' status
  // from a prior session was interrupted by the page closing — mark it as an
  // error so the UI doesn't show a forever spinner. The setState-in-effect is
  // intentional: localStorage isn't available during SSR, so lazy useState
  // init can't read it without a hydration mismatch.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Message[];
        const cleaned = parsed.map((m) =>
          m.role === "assistant" && m.status === "streaming"
            ? { ...m, status: "error" as const, error: "Interrupted by page reload." }
            : m,
        );
        // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time client hydration; see comment above
        setMessages(cleaned);
      }
    } catch {
      // Corrupted storage — start fresh; not worth surfacing.
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // Quota exceeded (data-URL images can be large) or storage disabled — drop.
    }
  }, [messages, hydrated]);

  function handleScroll(e: React.UIEvent<HTMLDivElement>) {
    const el = e.currentTarget;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 80;
  }

  useEffect(() => {
    if (!stickToBottomRef.current) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function ask(text: string, image?: { file: File; previewUrl: string }) {
    const userMsg: UserMessage = image
      ? { role: "user", text, image: { previewUrl: image.previewUrl } }
      : { role: "user", text };
    setMessages((m) => [
      ...m,
      userMsg,
      { role: "assistant", text: "", retrieved: [], status: "streaming" },
    ]);
    // The user just sent something — they want to see the new exchange.
    // Re-engage auto-scroll even if they had scrolled away earlier.
    stickToBottomRef.current = true;
    setStreaming(true);

    const ctl = new AbortController();
    abortRef.current = ctl;

    const stream: AsyncGenerator<StreamEvent> = image
      ? streamImageQuery(image.file, { text, signal: ctl.signal })
      : streamQuery(text, { signal: ctl.signal });

    try {
      for await (const ev of stream) {
        if (ev.type === "ocr") {
          patchLastUserImage((img) => ({ ...img, ocrText: ev.text }));
        } else if (ev.type === "retrieved") {
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

  function patchAssistant(fn: (m: AssistantMessage) => Message) {
    setMessages((all) => {
      const out = [...all];
      for (let i = out.length - 1; i >= 0; i--) {
        if (out[i].role === "assistant") {
          out[i] = fn(out[i] as AssistantMessage);
          break;
        }
      }
      return out;
    });
  }

  function patchLastUserImage(
    fn: (img: NonNullable<UserMessage["image"]>) => NonNullable<UserMessage["image"]>,
  ) {
    setMessages((all) => {
      const out = [...all];
      for (let i = out.length - 1; i >= 0; i--) {
        const m = out[i];
        if (m.role === "user" && m.image) {
          out[i] = { ...m, image: fn(m.image) };
          break;
        }
      }
      return out;
    });
  }

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file after clearing
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setImageError("Only image files are accepted.");
      return;
    }
    if (file.size > MAX_IMAGE_BYTES) {
      setImageError(`Image too large (${Math.round(file.size / 1024 / 1024)} MB; cap is 10 MB).`);
      return;
    }
    setImageError(null);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = () => setImagePreview(reader.result as string);
    reader.onerror = () => setImageError("Could not read the selected image.");
    reader.readAsDataURL(file);
  }

  function clearImage() {
    setImageFile(null);
    setImagePreview(null);
    setImageError(null);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (streaming) return;
    const text = input.trim();
    const file = imageFile;
    const preview = imagePreview;
    if (!file && !text) return;
    if (file && !preview) return; // FileReader hasn't finished yet

    setInput("");
    if (file && preview) {
      clearImage();
      void ask(text, { file, previewUrl: preview });
    } else {
      void ask(text);
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
  }

  function handleClear() {
    if (streaming) abortRef.current?.abort();
    setMessages([]);
  }

  const canSend = !streaming && hasDocuments && (input.trim().length > 0 || imageFile !== null);

  return (
    <div className="flex h-full flex-col">
      {messages.length > 0 && (
        <div className="mb-1 flex justify-end">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs text-muted-foreground"
            onClick={handleClear}
            aria-label="Clear chat history"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear chat
          </Button>
        </div>
      )}
      <div
        ref={scrollRootRef}
        onScroll={handleScroll}
        className="min-h-0 flex-1 overflow-y-auto px-1"
      >
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
      </div>

      <form onSubmit={handleSubmit} className="mt-3 space-y-2">
        {imagePreview && (
          <div className="flex items-center gap-3 rounded-md border bg-muted/40 p-2">
            {/* eslint-disable-next-line @next/next/no-img-element -- data URL, no remote optimization needed */}
            <img
              src={imagePreview}
              alt="Attached image preview"
              className="h-14 w-auto rounded object-contain"
            />
            <div className="min-w-0 flex-1 text-xs text-muted-foreground">
              <div className="truncate">{imageFile?.name}</div>
              <div>{imageFile ? `${Math.round(imageFile.size / 1024)} KB` : ""}</div>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0"
              onClick={clearImage}
              aria-label="Remove attached image"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        )}
        {imageError && (
          <p className="text-xs text-destructive" role="alert">
            {imageError}
          </p>
        )}

        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileSelect}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={streaming || !hasDocuments}
            aria-label="Attach image (OCR'd as the question)"
            title="Attach image — its text becomes the question"
          >
            <Paperclip className="h-4 w-4" />
          </Button>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={
              hasDocuments
                ? imageFile
                  ? "Optional context for the attached image…"
                  : "Ask about the uploaded paper(s)…"
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
            <Button type="submit" disabled={!canSend} aria-label="Send">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}

function MessageView({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex flex-col items-end gap-2">
        {message.image && (
          <div className="max-w-[85%]">
            {/* eslint-disable-next-line @next/next/no-img-element -- data URL stored client-side */}
            <img
              src={message.image.previewUrl}
              alt="Question image"
              className="max-h-64 rounded-lg border"
            />
          </div>
        )}
        {(message.text || message.image?.ocrText) && (
          <div className="max-w-[85%] space-y-1 rounded-lg bg-primary px-3.5 py-2.5 text-base font-medium text-primary-foreground">
            {message.text && <p className="whitespace-pre-wrap">{message.text}</p>}
            {message.image?.ocrText && (
              <p
                className={cn(
                  "whitespace-pre-wrap text-sm opacity-90",
                  message.text && "mt-1 border-t border-primary-foreground/20 pt-1",
                )}
              >
                <span className="mr-1 text-xs font-semibold uppercase tracking-wide opacity-70">
                  From image
                </span>
                {message.image.ocrText}
              </p>
            )}
          </div>
        )}
        {message.image && !message.image.ocrText && (
          <p className="text-xs italic text-muted-foreground">
            <Loader2 className="mr-1 inline h-3 w-3 animate-spin" />
            Transcribing image…
          </p>
        )}
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
          ? "Ask a question about your uploaded paper. Type, or attach an image of text — both work."
          : "Upload a PDF on the left to get started. Born-digital arXiv-style PDFs work best."}
      </p>
    </div>
  );
}
