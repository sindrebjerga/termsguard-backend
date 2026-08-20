import "dotenv/config";
import express from "express";
import cors from "cors";
import Anthropic from "@anthropic-ai/sdk";

const PORT = process.env.PORT || 8787;
// Haiku by default for speed (this runs in a browser popup — users won't
// wait around for a slow model). Both models occasionally emit a malformed
// field in this multi-turn tool-use flow, so we validate the output and
// retry once with the stronger model rather than always paying Sonnet's
// latency up front.
const FAST_MODEL = process.env.FINEPRINT_MODEL || "claude-haiku-4-5-20251001";
const FALLBACK_MODEL = process.env.FINEPRINT_FALLBACK_MODEL || "claude-sonnet-5";

if (!process.env.ANTHROPIC_API_KEY) {
  console.error("Missing ANTHROPIC_API_KEY. Copy .env.example to .env and set it.");
  process.exit(1);
}

const anthropic = new Anthropic();
const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

function isSafeUrl(urlStr) {
  try {
    const u = new URL(urlStr);
    if (!["http:", "https:"].includes(u.protocol)) return false;
    const host = u.hostname.toLowerCase();
    if (host === "localhost" || host === "127.0.0.1" || host === "::1") return false;
    if (/^10\./.test(host)) return false;
    if (/^192\.168\./.test(host)) return false;
    if (/^172\.(1[6-9]|2\d|3[01])\./.test(host)) return false;
    if (host === "169.254.169.254") return false;
    return true;
  } catch {
    return false;
  }
}

function htmlToText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/(p|div|li|h[1-6]|tr)>/gi, "\n")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n+/g, "\n")
    .trim();
}

async function fetchLinkedText(url, timeoutMs = 5000) {
  if (!isSafeUrl(url)) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      redirect: "follow",
      headers: { "User-Agent": "Mozilla/5.0 (FineprintGuardian/0.1)" },
    });
    if (!res.ok) return null;
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("text/html") && !contentType.includes("text/plain")) return null;
    const html = await res.text();
    const text = htmlToText(html).slice(0, 12000);
    return text.trim() ? text : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

const MAX_LINK_READS = 2;

const SYSTEM_PROMPT = `You are a consumer-protection assistant helping a user understand a webpage they're about to act on — a checkout, a signup form, a subscription page, or a terms/contract page. The page can be in any language. Always write your findings in English regardless of the page's language, except for short direct quotes.

You are given the visible page text and a list of links found on the page (link text -> URL). Some of those links may lead to the actual terms, conditions, cancellation policy, or pricing details that matter most for this page — regardless of what they're labeled. Labels vary a lot by site and language: "Terms", "Terms of Service", "Vilkår", "Avtalevilkår", "Conditions générales", "AGB", "Read more about our policies", "Angrerett", etc. Use your judgment on the link text to decide which ones are actually likely to matter here, and ignore irrelevant links (navigation, social media, unrelated articles, login, etc).

Use the read_link tool to fetch and read up to ${MAX_LINK_READS} of the most relevant links before finalizing your analysis, if any look relevant. It's fine to call report_findings directly with zero read_link calls if no links on the page look relevant, or if the page text alone is already a complete, self-contained agreement.

The whole point of this tool is to save the user from reading everything themselves. Be ruthless about brevity and prioritization — a short, skimmable result that surfaces what actually matters beats a thorough one that lists everything. When in doubt, leave it out.

Once you've gathered what you need, you have three jobs, all required, delivered via report_findings:

1. summary: exactly ONE short plain-English sentence (under 25 words) giving the single most important takeaway — what kind of page this is and the headline verdict. Not a recap of the page. E.g. "This is a 12-month gym membership with an auto-renewal clause and a strict cancellation window." or "This is a straightforward pricing page with no concerning terms." This field must never be empty.

2. key_facts: AT MOST 5 short bullets — only the handful of facts someone would actually want to know in 10 seconds to make the decision (cost, commitment length, what the real choices are, e.g. data tiers if that's what's being chosen). Skip minor administrative details (invoice format, support hours, routine account requirements) unless real money or commitment is involved. This is a highlight reel, not a transcript.

3. flags: AT MOST 4 flags, most important first, in these categories: auto_renewal, hidden_fees, cancellation_difficulty, refund_policy, arbitration_clause, data_sharing, other. Only flag things a reasonably careful consumer would genuinely want a heads-up about before agreeing. Skip routine, expected, or trivial details even if technically present in the text (e.g. a standard credit check, a small paper-invoice surcharge, normal shipping terms are NOT flags).

Rules:
- Only flag things actually present in the text you were given (page text and/or fetched linked pages). Do not invent or assume issues that aren't stated.
- If the page (and anything you read) is low-risk, return an empty flags array and overall_risk "low" — but summary and key_facts must still be filled in.
- Each flag's explanation must be ONE short plain-English sentence, written for a non-lawyer.
- Call report_findings exactly once, as your final action.`;

const READ_LINK_TOOL = {
  name: "read_link",
  description: "Fetch and read the text content of a link found on the page, to check whether it contains terms, conditions, cancellation policy, pricing, or other details relevant to this agreement. Works in any language.",
  input_schema: {
    type: "object",
    properties: {
      url: {
        type: "string",
        description: "The absolute URL to fetch, taken exactly from the list of links provided.",
      },
    },
    required: ["url"],
  },
};

const REPORT_TOOL = {
  name: "report_findings",
  description: "Report the fine-print risk findings for this page.",
  input_schema: {
    type: "object",
    properties: {
      overall_risk: { type: "string", enum: ["low", "medium", "high"] },
      summary: {
        type: "string",
        description: "Exactly one short plain-English sentence (under 25 words) — the single most important takeaway, not a recap.",
      },
      key_facts: {
        type: "array",
        description: "At most 5 short bullets — only the facts that actually matter for the decision (cost, commitment length, the real choices on offer). Not a full recap of the page.",
        items: { type: "string" },
        maxItems: 5,
      },
      flags: {
        type: "array",
        description: "At most 4 flags, most important first. Only genuinely noteworthy issues — skip routine/trivial details.",
        maxItems: 4,
        items: {
          type: "object",
          properties: {
            category: {
              type: "string",
              enum: [
                "auto_renewal",
                "hidden_fees",
                "cancellation_difficulty",
                "refund_policy",
                "arbitration_clause",
                "data_sharing",
                "other",
              ],
            },
            severity: { type: "string", enum: ["low", "medium", "high"] },
            explanation: { type: "string" },
            quote: {
              type: "string",
              description: "A short exact quote from the page text that triggered this flag, if available.",
            },
          },
          required: ["category", "severity", "explanation"],
        },
      },
    },
    required: ["overall_risk", "summary", "key_facts", "flags"],
  },
};

// Defensive: tool-use output is schema-guided, not schema-enforced, so a
// model can still occasionally return e.g. key_facts as a malformed string,
// or leak bits of its own tool-call structure (e.g. "<category>other</category>")
// as stray list entries. Normalize before this ever reaches the UI, and drop
// anything that looks like leaked internal structure rather than try to clean it.
const LEAKED_FIELD_MARKERS = /<\/?(category|severity|explanation|quote|item)>/i;

function normalizeStringList(value) {
  let candidates;
  if (Array.isArray(value)) {
    candidates = value.filter((v) => typeof v === "string" && v.trim());
  } else if (typeof value === "string") {
    const itemMatches = [...value.matchAll(/<item>([\s\S]*?)<\/item>/gi)].map((m) => m[1]);
    if (itemMatches.length) {
      candidates = itemMatches;
    } else {
      const lines = value.split(/\n+/).filter((s) => s.trim());
      candidates = lines.length > 1 ? lines : [value];
    }
  } else {
    candidates = [];
  }

  const cleaned = [];
  for (const raw of candidates) {
    if (LEAKED_FIELD_MARKERS.test(raw)) continue;
    const text = raw.replace(/<\/?[^>]+>/g, "").trim();
    if (text) cleaned.push(text);
  }
  return cleaned;
}

function looksCorrupted(rawResult) {
  const suspicious = (v) => {
    if (typeof v === "string") return LEAKED_FIELD_MARKERS.test(v);
    if (Array.isArray(v)) return v.some((x) => typeof x === "string" && LEAKED_FIELD_MARKERS.test(x));
    return false;
  };
  return suspicious(rawResult?.key_facts) || suspicious(rawResult?.flags) || suspicious(rawResult?.summary);
}

function normalizeResult(result) {
  const out = { ...result };
  out.overall_risk = ["low", "medium", "high"].includes(out.overall_risk) ? out.overall_risk : "medium";
  out.key_facts = normalizeStringList(out.key_facts).slice(0, 5);

  out.summary = typeof out.summary === "string" ? out.summary.trim() : "";
  if (!out.summary) {
    const riskWord = { low: "low-risk", medium: "worth a closer look", high: "high-risk" }[out.overall_risk];
    out.summary = `This page appears ${riskWord} based on the scan.`;
  }

  const validCategories = [
    "auto_renewal", "hidden_fees", "cancellation_difficulty",
    "refund_policy", "arbitration_clause", "data_sharing", "other",
  ];
  out.flags = Array.isArray(out.flags)
    ? out.flags
        .filter((f) => f && typeof f.explanation === "string" && !LEAKED_FIELD_MARKERS.test(f.explanation))
        .map((f) => ({
          category: validCategories.includes(f.category) ? f.category : "other",
          severity: ["low", "medium", "high"].includes(f.severity) ? f.severity : "medium",
          explanation: f.explanation,
          quote: typeof f.quote === "string" ? f.quote : undefined,
        }))
        .slice(0, 4)
    : [];

  return out;
}

function buildInitialPrompt(text, url, pageLinks) {
  const links = Array.isArray(pageLinks) ? pageLinks.slice(0, 60) : [];
  const linksBlock = links.length
    ? `\n\nLinks found on this page (link text -> URL):\n${links.map((l) => `- "${l.text}" -> ${l.href}`).join("\n")}`
    : "";
  return `Page URL: ${url || "unknown"}\n\nPage text:\n${text}${linksBlock}`;
}

async function runScan(text, url, pageLinks, model) {
  const truncated = text.slice(0, 12000);
  const messages = [{ role: "user", content: buildInitialPrompt(truncated, url, pageLinks) }];

  const sourcesRead = [];
  const sourcesFailed = [];
  let linkReadsUsed = 0;

  for (let turn = 0; turn < MAX_LINK_READS + 3; turn++) {
    const response = await anthropic.messages.create({
      model,
      max_tokens: 2000,
      system: SYSTEM_PROMPT,
      tools: [READ_LINK_TOOL, REPORT_TOOL],
      tool_choice: { type: "auto" },
      messages,
    });

    const toolUses = response.content.filter((b) => b.type === "tool_use");
    const reportUse = toolUses.find((t) => t.name === "report_findings");
    if (reportUse) {
      const result = normalizeResult(reportUse.input);
      result.sources_read = sourcesRead;
      result.sources_failed = sourcesFailed;
      return { result, raw: reportUse.input };
    }

    const readUses = toolUses.filter((t) => t.name === "read_link");
    if (readUses.length > 0) {
      messages.push({ role: "assistant", content: response.content });
      const resultBlocks = [];
      for (const use of readUses) {
        let content;
        if (linkReadsUsed >= MAX_LINK_READS) {
          content = "Link-reading budget used up for this scan. Finalize your analysis with what you have.";
        } else {
          linkReadsUsed++;
          const linkText = await fetchLinkedText(use.input.url);
          if (linkText) {
            sourcesRead.push(use.input.url);
            content = linkText;
          } else {
            sourcesFailed.push(use.input.url);
            content = "Could not fetch this URL (blocked, not readable HTML, or timed out). Continue without it.";
          }
        }
        resultBlocks.push({ type: "tool_result", tool_use_id: use.id, content });
      }
      messages.push({ role: "user", content: resultBlocks });
      continue;
    }

    // No tool use at all this turn — nudge the model to finalize.
    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: "Please finalize your analysis now by calling report_findings." });
  }

  throw new Error("Model did not return structured findings after tool use.");
}

app.post("/scan", async (req, res) => {
  const { text, url, pageLinks } = req.body || {};
  if (!text || typeof text !== "string" || !text.trim()) {
    return res.status(400).json({ error: "Missing page text." });
  }

  try {
    let { result, raw } = await runScan(text, url, pageLinks, FAST_MODEL);
    if (looksCorrupted(raw)) {
      console.warn("Fast model produced malformed output, retrying with fallback model");
      ({ result } = await runScan(text, url, pageLinks, FALLBACK_MODEL));
    }
    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Scan failed. Try again." });
  }
});

app.get("/health", (_req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`TermsGuard backend listening on http://localhost:${PORT}`);
});
