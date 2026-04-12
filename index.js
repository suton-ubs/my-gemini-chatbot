import dotenv from "dotenv";
import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import express from "express";
import cors from "cors";
import multer from "multer";
import { GoogleGenAI, createPartFromUri } from "@google/genai";

dotenv.config();

const ROOT_DIR = process.cwd();
const UPLOADS_DIR = path.join(ROOT_DIR, "uploads");
const LOGS_DIR = path.join(ROOT_DIR, "logs");
const PERSONALITIES_FILE_PATH = path.join(ROOT_DIR, "personalities.json");

const JSON_BODY_LIMIT = "10mb";
const DEFAULT_MAX_UPLOAD_SIZE_MB = 25;
const GEMINI_TIMEOUT_MS = 30_000;

const CHAT_ROLES = new Set(["system", "user", "model"]);

const env = buildEnv();
const geminiClient = new GoogleGenAI({ apiKey: env.geminiApiKey });
let personalityCatalog = new Map();

// Domain: Main entry and startup orchestration.
// Starts runtime dependencies, composes app, then attaches graceful shutdown hooks.

/**
 * Bootstraps runtime directories, starts HTTP server, and binds shutdown hooks.
 *
 * @returns {Promise<void>}
 */
async function bootstrap() {
  await ensureRuntimeDirectories();
  await cleanStaleUploads();
  personalityCatalog = await loadPersonalityCatalog();

  const app = createApp();
  const server = app.listen(env.port, () => {
    console.log(`My Gemini Chatbot server running on port ${env.port}`);
  });

  setupGracefulShutdown(server);
}

/**
 * Ensures runtime directories exist before handling requests.
 *
 * @returns {Promise<void>}
 */
async function ensureRuntimeDirectories() {
  await fs.mkdir(UPLOADS_DIR, { recursive: true });
  await fs.mkdir(LOGS_DIR, { recursive: true });
}

/**
 * Removes stale files left behind in the uploads directory.
 *
 * @returns {Promise<void>}
 */
async function cleanStaleUploads() {
  const files = await fs.readdir(UPLOADS_DIR);

  await Promise.all(
    files.map((fileName) => fs.unlink(path.join(UPLOADS_DIR, fileName)).catch(() => {}))
  );
}

/**
 * Loads and validates personalities from personalities.json.
 *
 * @returns {Promise<Map<string, {
 * id: string,
 * title: string,
 * shortDescription: string,
 * generationConfig: {temperature: number, topK: number, topP: number, maxOutputTokens: number},
 * systemInstruction: string
 * }>>} Map keyed by personality id.
 * @throws {Error} When JSON format or required fields are invalid.
 */
async function loadPersonalityCatalog() {
  let rawJson = "";

  try {
    rawJson = await fs.readFile(PERSONALITIES_FILE_PATH, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") {
      return new Map();
    }

    throw new Error(`Failed to read personalities.json: ${error.message}`);
  }

  let parsed = null;

  try {
    parsed = JSON.parse(rawJson);
  } catch (error) {
    throw new Error(`personalities.json is not valid JSON: ${error.message}`);
  }

  assertCondition(Array.isArray(parsed), "personalities.json must be an array.");

  const catalog = new Map();

  for (const [index, item] of parsed.entries()) {
    assertCondition(item && typeof item === "object" && !Array.isArray(item), `Personality at index ${index} must be an object.`);

    const id = optionalString(item.id);
    assertCondition(Boolean(id), `Personality at index ${index} is missing \"id\".`);
    assertCondition(!catalog.has(id), `Duplicate personality id \"${id}\" found in personalities.json.`);

    const config = item.config;
    assertCondition(config && typeof config === "object" && !Array.isArray(config), `Personality \"${id}\" must include object \"config\".`);

    const temperature = Number(config.gemini_temperature);
    const topK = Number(config.gemini_top_k);
    const topP = Number(config.gemini_top_p);
    const maxOutputTokens = Number(config.gemini_max_tokens);
    const systemInstruction = optionalString(config.system_instruction);

    assertCondition(Number.isFinite(temperature) && temperature >= 0 && temperature <= 2, `Personality \"${id}\" has invalid config.gemini_temperature (must be between 0 and 2).`);
    assertCondition(Number.isFinite(topK) && topK > 0, `Personality \"${id}\" has invalid config.gemini_top_k (must be greater than 0).`);
    assertCondition(Number.isFinite(topP) && topP > 0 && topP <= 1, `Personality \"${id}\" has invalid config.gemini_top_p (must be > 0 and <= 1).`);
    assertCondition(Number.isFinite(maxOutputTokens) && maxOutputTokens > 0, `Personality \"${id}\" has invalid config.gemini_max_tokens (must be greater than 0).`);

    catalog.set(id, {
      id,
      title: optionalString(item.title),
      shortDescription: optionalString(item.short_description),
      generationConfig: {
        temperature,
        topK,
        topP,
        maxOutputTokens,
      },
      systemInstruction,
    });
  }

  return catalog;
}

// Domain: App composition and lifecycle wiring.
// Defines Express app graph and graceful termination behavior.

/**
 * Creates the Express app with middleware, routes, and centralized error mapping.
 *
 * @returns {import("express").Express} Configured Express application.
 */
function createApp() {
  const app = express();

  // Hide framework signature to reduce passive fingerprinting.
  app.disable("x-powered-by");

  // Request id helps us trace one request across logs and responses.
  app.use((req, res, next) => {
    const incomingRequestId = optionalString(req.header("x-request-id"));
    req.requestId = incomingRequestId || createRequestId();
    res.setHeader("x-request-id", req.requestId);
    next();
  });

  // CORS is restricted to explicit origins from .env.
  app.use(cors(buildCorsOptions()));

  // Explicit body limits protect memory from unexpectedly large payloads.
  app.use(express.json({ limit: JSON_BODY_LIMIT }));
  app.use(express.urlencoded({ extended: true, limit: JSON_BODY_LIMIT }));

  // Static public page is only for manual testing; API works independently.
  app.use(express.static(path.join(ROOT_DIR, "public")));

  app.get("/health", (req, res) => {
    return res.json({
      status: "ok",
      requestId: req.requestId,
      uptimeSeconds: Math.round(process.uptime()),
      timestamp: new Date().toISOString(),
    });
  });

  app.get("/chat-meta", (req, res) => {
    const personalities = Array.from(personalityCatalog.values()).map((item) => ({
      id: item.id,
      title: item.title || item.id,
      shortDescription: item.shortDescription || "",
    }));

    return res.json({
      status: "success",
      requestId: req.requestId,
      defaultModel: env.geminiDefaultModel,
      allowedModels: env.geminiAllowedModels,
      allowedUploadExtensions: getAllowedExtensionsForRoute("chat"),
      personalities,
    });
  });

  app.post("/generate-text", requireAccessCode, generateText);
  app.post("/chat", requireAccessCode, uploadChatFile, chat);
  app.post("/generate-from-image", requireAccessCode, uploadImage, generateFromImage);
  app.post("/generate-from-document", requireAccessCode, uploadDocument, generateFromDocument);
  app.post("/generate-from-audio", requireAccessCode, uploadAudio, generateFromAudio);

  app.use((error, req, res, next) => {
    if (res.headersSent) {
      return next(error);
    }

    const requestId = req.requestId || createRequestId();

    if (error instanceof multer.MulterError) {
      return res.status(400).json(buildErrorResponse(requestId, `Upload failed: ${error.message}`));
    }

    // Centralized mapping keeps error shape predictable for every endpoint.
    const statusCode = error.statusCode || error.status || 500;

    if (statusCode >= 500) {
      console.error(`[${requestId}] Unexpected error: ${error.message}`);
    }

    const baseMessage = optionalString(error.message) || "Unexpected server error.";
    const message = statusCode >= 500 ? `Processing failed: ${baseMessage}` : baseMessage;

    return res.status(statusCode).json(buildErrorResponse(requestId, message));
  });

  return app;
}

/**
 * Registers graceful shutdown handlers for process termination signals.
 *
 * @param {import("node:http").Server} server - Active HTTP server instance.
 * @returns {void}
 */
function setupGracefulShutdown(server) {
  let shuttingDown = false;

  /**
   * Closes the server and exits once shutdown starts.
   *
   * @param {"SIGINT" | "SIGTERM"} signal - Signal that triggered shutdown.
   * @returns {void}
   */
  const shutdown = (signal) => {
    if (shuttingDown) {
      return;
    }

    shuttingDown = true;
    // Stop accepting new connections and let active work finish cleanly.
    console.log(`Received ${signal}. Shutting down server...`);

    server.close((error) => {
      if (error) {
        console.error("Error during shutdown:", error.message);
        process.exit(1);
      }

      console.log("Server stopped cleanly.");
      process.exit(0);
    });

    setTimeout(() => {
      console.error("Forced shutdown after timeout.");
      process.exit(1);
    }, 10_000).unref();
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));
}

// Domain: Catalog and model resolution.
// Maps request-level personality/model parameters to validated runtime config.

/**
 * Resolves one personality by id from the loaded catalog.
 *
 * @param {string} personalityId - Personality id from request payload.
 * @returns {{
 * id: string,
 * title: string,
 * shortDescription: string,
 * generationConfig: {temperature: number, topK: number, topP: number, maxOutputTokens: number},
 * systemInstruction: string
 * }} Personality configuration.
 * @throws {Error & {statusCode: number}} When id is unknown.
 */
function resolvePersonality(personalityId) {
  const personality = personalityCatalog.get(personalityId);

  if (!personality) {
    throw createHttpError(400, `Field \"personalityId\" must exist in personalities.json. Unknown id: \"${personalityId}\".`);
  }

  return personality;
}

/**
 * Generates a request id used for tracing logs and responses.
 *
 * @returns {string} UUID (when available) or timestamp-based fallback id.
 */
function createRequestId() {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Sanitizes an uploaded file name to avoid unsafe characters.
 *
 * @param {string} fileName - Original file name.
 * @returns {string} Safe basename for storage.
 */
function sanitizeFileName(fileName) {
  return path.basename(fileName || "upload.bin").replace(/[^a-zA-Z0-9._-]/g, "_");
}

/**
 * Extracts a lowercase extension (without dot) from a file name.
 *
 * @param {string} fileName - File name to inspect.
 * @returns {string} Extension without leading dot.
 */
function getFileExtension(fileName) {
  const extension = path.extname(fileName || "").toLowerCase();
  return extension.startsWith(".") ? extension.slice(1) : extension;
}

/**
 * Resolves a requested model and validates it against the allowlist.
 *
 * @param {string} requestedModel - Optional model from request payload.
 * @returns {string} Valid Gemini model id.
 * @throws {Error & {statusCode: number}} When model is not allowed.
 */
function resolveGeminiModel(requestedModel) {
  const model = optionalString(requestedModel) || env.geminiDefaultModel;

  if (!env.geminiAllowedModels.includes(model)) {
    throw createHttpError(400, `Gemini model "${model}" is not allowed.`);
  }

  return model;
}

/**
 * Builds shared Gemini generation config for all endpoints.
 *
 * @param {{
 * temperature?: number,
 * topK?: number,
 * topP?: number,
 * maxOutputTokens?: number,
 * systemInstruction?: string
 * }} [overrides] - Optional generation overrides.
 * @returns {{temperature: number, topK: number, topP: number, maxOutputTokens: number, systemInstruction?: string}} Generation config.
 */
function buildGenerationConfig(overrides = {}) {
  const config = {
    temperature: Number.isFinite(overrides.temperature) ? overrides.temperature : env.geminiTemperature,
    topK: Number.isFinite(overrides.topK) ? overrides.topK : env.geminiTopK,
    topP: Number.isFinite(overrides.topP) ? overrides.topP : env.geminiTopP,
    maxOutputTokens: Number.isFinite(overrides.maxOutputTokens) ? overrides.maxOutputTokens : env.geminiMaxTokens,
  };

  const normalizedInstruction = optionalString(overrides.systemInstruction);
  if (normalizedInstruction) {
    config.systemInstruction = normalizedInstruction;
  }

  return config;
}

// Domain: Upstream interaction primitives.
// Wraps Gemini calls with timeout and response text extraction.

/**
 * Resolves a promise with timeout protection.
 *
 * @template T
 * @param {Promise<T>} promise - Promise to wrap.
 * @param {number} timeoutMs - Timeout in milliseconds.
 * @param {string} label - Name used in timeout error message.
 * @returns {Promise<T>} Promise result when completed in time.
 * @throws {Error} When timeout is reached.
 */
async function withTimeout(promise, timeoutMs, label) {
  let timer = null;

  const timeoutPromise = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs} ms.`));
    }, timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

/**
 * Extracts text output from multiple Gemini response shapes.
 *
 * @param {any} response - Raw Gemini API response.
 * @returns {string} Best-effort plain text output.
 */
function extractGeminiText(response) {
  if (typeof response?.text === "string" && response.text.trim().length > 0) {
    return response.text;
  }

  const candidateParts = response?.candidates?.[0]?.content?.parts || [];
  return candidateParts
    .map((part) => part?.text)
    .filter(Boolean)
    .join("\n")
    .trim();
}

  /**
   * Extracts plain text from a chat message object.
   *
   * @param {any} message - Message object that may use text/content/parts fields.
   * @returns {string} Extracted and trimmed text, or empty string.
   */
function extractMessageText(message) {
  if (typeof message?.text === "string") {
    return message.text.trim();
  }

  if (typeof message?.content === "string") {
    return message.content.trim();
  }

  if (Array.isArray(message?.parts)) {
    return message.parts
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        if (typeof part?.text === "string") {
          return part.text;
        }

        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }

  if (Array.isArray(message?.content)) {
    return message.content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        if (typeof part?.text === "string") {
          return part.text;
        }

        return "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }

  return "";
}

// Domain: Chat payload normalization.
// Parses and validates message arrays before mapping into Gemini chat contents.

/**
 * Parses chat messages from either JSON requests or multipart text fields.
 *
 * @param {unknown} rawMessages - Raw incoming messages value.
 * @returns {unknown} Parsed messages value.
 * @throws {Error & {statusCode: number}} When multipart messages are invalid JSON.
 */
function parseChatMessagesInput(rawMessages) {
  if (typeof rawMessages !== "string") {
    return rawMessages;
  }

  const normalizedRawMessages = rawMessages.trim();
  if (!normalizedRawMessages) {
    throw createHttpError(400, 'Field "messages" must not be empty.');
  }

  try {
    return JSON.parse(normalizedRawMessages);
  } catch (_error) {
    throw createHttpError(400, 'Field "messages" must be valid JSON when sent as multipart/form-data.');
  }
}

/**
 * Validates and normalizes raw chat messages into strict internal shape.
 *
 * @param {any} rawMessages - Incoming messages payload.
 * @returns {{role: "system" | "user" | "model", text: string}[]} Normalized messages.
 * @throws {Error & {statusCode: number}} When payload is invalid.
 */
function normalizeChatMessages(rawMessages) {
  if (!Array.isArray(rawMessages)) {
    throw createHttpError(400, 'Field "messages" must be an array.');
  }

  if (rawMessages.length === 0) {
    throw createHttpError(400, 'Field "messages" must include at least 1 message.');
  }

  const normalizedMessages = rawMessages.map((message, index) => {
    if (!message || typeof message !== "object" || Array.isArray(message)) {
      throw createHttpError(400, `Message at index ${index} must be an object.`);
    }

    const role = optionalString(message.role).toLowerCase();
    if (!CHAT_ROLES.has(role)) {
      throw createHttpError(400, `Message at index ${index} has invalid role "${message.role}". Allowed roles: system, user, model.`);
    }

    const text = extractMessageText(message);
    if (!text) {
      throw createHttpError(400, `Message at index ${index} must include text in "text", "content", or "parts".`);
    }

    return {
      role,
      text,
    };
  });

  const hasUserRole = normalizedMessages.some((message) => message.role === "user");
  if (!hasUserRole) {
    throw createHttpError(400, 'Field "messages" must include at least one message with role "user".');
  }

  const lastMessage = normalizedMessages[normalizedMessages.length - 1];
  // Last message must be from the user so Gemini knows it should generate a reply.
  if (lastMessage.role !== "user") {
    throw createHttpError(400, 'Last message role must be "user" so the model can respond.');
  }

  return normalizedMessages;
}

/**
 * Converts normalized chat messages into Gemini contents and system instruction.
 *
 * @param {{role: "system" | "user" | "model", text: string}[]} normalizedMessages - Validated chat messages.
 * @param {string} [fallbackSystemInstruction] - Default system instruction when payload has no system message.
 * @returns {{contents: {role: "user" | "model", parts: {text: string}[]}[], systemInstruction: string}} Gemini-ready payload.
 * @throws {Error & {statusCode: number}} When all messages are system-only.
 */
function buildGeminiChatPayload(normalizedMessages, fallbackSystemInstruction = env.systemInstruction) {
  const payloadSystemInstructions = [];
  const contents = [];

  for (const message of normalizedMessages) {
    // Keep system messages separate from conversation contents for Gemini config.
    if (message.role === "system") {
      payloadSystemInstructions.push(message.text);
      continue;
    }

    contents.push({
      role: message.role,
      parts: [{ text: message.text }],
    });
  }

  if (contents.length === 0) {
    throw createHttpError(400, "Chat messages must include at least one non-system message.");
  }

  // Payload-level system messages override fallback instruction for this request.
  const payloadInstruction = payloadSystemInstructions.join("\n\n").trim();
  const systemInstruction = payloadInstruction || optionalString(fallbackSystemInstruction);

  return {
    contents,
    systemInstruction,
  };
}

/**
 * Attaches an uploaded file part to the latest user message in chat contents.
 *
 * @param {{role: "user" | "model", parts: {text?: string}[]}[]} chatContents - Gemini chat contents.
 * @param {{uri: string, mimeType: string} | null} uploadedGeminiFile - Uploaded file reference.
 * @returns {{role: "user" | "model", parts: object[]}[]} Updated chat contents.
 */
function attachUploadedFileToChatContents(chatContents, uploadedGeminiFile) {
  if (!uploadedGeminiFile) {
    return chatContents;
  }

  const filePart = createPartFromUri(uploadedGeminiFile.uri, uploadedGeminiFile.mimeType);
  const nextContents = chatContents.map((item) => ({
    ...item,
    parts: Array.isArray(item.parts) ? [...item.parts] : [],
  }));

  for (let index = nextContents.length - 1; index >= 0; index -= 1) {
    if (nextContents[index].role === "user") {
      nextContents[index].parts.push(filePart);
      return nextContents;
    }
  }

  return [...nextContents, { role: "user", parts: [filePart] }];
}

// Domain: Request logging.
// Builds safe JSON logs for every endpoint request/response lifecycle.

/**
 * Safely serializes unknown values for structured logging.
 *
 * @param {any} value - Value to serialize.
 * @returns {any} Serializable clone or fallback note object.
 */
function toLogSafeValue(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (_error) {
    return { note: "Non-serializable value omitted from log." };
  }
}

/**
 * Builds a timestamped log file name for an endpoint.
 *
 * @param {string} endpointName - Endpoint key used in file naming.
 * @returns {string} Sanitized file name.
 */
function makeLogFileName(endpointName) {
  const safeEndpoint = (endpointName || "endpoint").replace(/[^a-zA-Z0-9_-]/g, "_");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  return `${timestamp}-${safeEndpoint}.json`;
}

/**
 * Writes one JSON log entry to the logs directory.
 *
 * @param {object} entry - Log entry payload.
 * @returns {Promise<string>} Written log file name.
 */
async function writeRequestLog(entry) {
  const fileName = makeLogFileName(entry.endpoint);
  const filePath = path.join(LOGS_DIR, fileName);

  await fs.writeFile(filePath, JSON.stringify(entry, null, 2), "utf8");
  return fileName;
}

/**
 * Writes a log entry and suppresses logging failures.
 *
 * @param {object} entry - Log entry payload.
 * @returns {Promise<string | null>} File name when successful, otherwise null.
 */
async function safeWriteRequestLog(entry) {
  try {
    return await writeRequestLog(entry);
  } catch (error) {
    console.error(`[${entry.requestId}] Failed to write log file: ${error.message}`);
    return null;
  }
}

// Domain: File lifecycle management.
// Handles temporary upload cleanup and Gemini Files API upload/delete flow.

/**
 * Removes a file if it exists.
 *
 * @param {string | undefined | null} filePath - Absolute path to delete.
 * @returns {Promise<void>}
 */
async function removeFileIfExists(filePath) {
  if (!filePath) {
    return;
  }

  try {
    await fs.unlink(filePath);
  } catch (error) {
    if (error.code !== "ENOENT") {
      console.error(`Failed to delete temp upload: ${error.message}`);
    }
  }
}

/**
 * Uploads a local multer file to Gemini Files API.
 *
 * @param {{path: string, mimetype?: string, originalname?: string}} file - Multer file object.
 * @returns {Promise<{name: string, uri: string, mimeType: string}>} Uploaded Gemini file reference.
 * @throws {Error & {statusCode: number}} When response is missing required fields.
 */
async function uploadFileToGemini(file) {
  const uploadedFile = await geminiClient.files.upload({
    file: file.path,
    config: {
      mimeType: optionalString(file.mimetype) || undefined,
      displayName: optionalString(file.originalname) || undefined,
    },
  });

  const name = optionalString(uploadedFile?.name);
  const uri = optionalString(uploadedFile?.uri);
  const mimeType = optionalString(uploadedFile?.mimeType);

  if (!name || !uri || !mimeType) {
    throw createHttpError(500, "Gemini file upload did not return a usable file reference.");
  }

  return {
    name,
    uri,
    mimeType,
  };
}

/**
 * Deletes a Gemini file and ignores cleanup failures.
 *
 * @param {string | undefined | null} fileName - Gemini file resource name.
 * @param {string} requestId - Request id for error log context.
 * @returns {Promise<void>}
 */
async function safeDeleteGeminiFile(fileName, requestId) {
  if (!fileName) {
    return;
  }

  try {
    await geminiClient.files.delete({ name: fileName });
  } catch (error) {
    console.error(`[${requestId}] Failed to delete Gemini file: ${error.message}`);
  }
}

/**
 * Builds the standardized API success response envelope.
 *
 * @param {{requestId: string, model: string, output: string}} params - Response payload values.
 * @returns {{status: "success", requestId: string, model: string, output: string}} Success response.
 */
function buildSuccessResponse({ requestId, model, output }) {
  return {
    status: "success",
    requestId,
    model,
    output,
  };
}

/**
 * Builds the standardized API error response envelope.
 *
 * @param {string} requestId - Request id for traceability.
 * @param {string} message - Error message to return.
 * @returns {{status: "error", requestId: string, error: string}} Error response.
 */
function buildErrorResponse(requestId, message) {
  return {
    status: "error",
    requestId,
    error: message,
  };
}

// Domain: Upload policy and middleware.
// Enforces route extension allowlists and configures multer per endpoint.

const routeExtensionMap = {
  image: ["png", "bmp", "jpg", "jpeg", "webp"],
  document: ["pdf", "txt"],
  audio: ["mp3", "wav"],
};

/**
 * Intersects route-specific extension rules with global extension allowlist.
 *
 * @param {"image" | "document" | "audio" | "chat"} routeType - Route category.
 * @returns {string[]} Effective allowed extensions.
 */
function getAllowedExtensionsForRoute(routeType) {
  if (routeType === "chat") {
    return [...new Set(env.allowedUploadExtensions)];
  }

  // Route-level allowlist must also exist in the global env allowlist.
  const globalAllowed = new Set(env.allowedUploadExtensions);
  return routeExtensionMap[routeType].filter((extension) => globalAllowed.has(extension));
}

/**
 * Creates a multer single-file middleware with route-level extension validation.
 *
 * @param {string} fieldName - Multipart field name.
 * @param {"image" | "document" | "audio" | "chat"} routeType - Route category for validation.
 * @returns {import("express").RequestHandler} Multer middleware.
 */
function buildUploadMiddleware(fieldName, routeType) {
  const allowedForRoute = getAllowedExtensionsForRoute(routeType);

  const storage = multer.diskStorage({
    destination: (_req, _file, callback) => {
      callback(null, UPLOADS_DIR);
    },
    filename: (_req, file, callback) => {
      // Add timestamp + random suffix so simultaneous uploads do not collide.
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      const randomSuffix = Math.random().toString(36).slice(2, 9);
      const safeName = sanitizeFileName(file.originalname || "upload.bin");
      callback(null, `${timestamp}-${randomSuffix}-${safeName}`);
    },
  });

  return multer({
    storage,
    limits: { fileSize: env.maxUploadSizeBytes },
    fileFilter: (_req, file, callback) => {
      // Validate extension before file processing to reduce risky input handling.
      const extension = getFileExtension(file.originalname);
      if (!allowedForRoute.includes(extension)) {
        const allowedText = allowedForRoute.length > 0 ? allowedForRoute.join(", ") : "(none configured)";
        return callback(createHttpError(400, `Unsupported file extension ".${extension}" for ${routeType}. Allowed: ${allowedText}.`));
      }

      return callback(null, true);
    },
  }).single(fieldName);
}

const uploadImage = buildUploadMiddleware("image", "image");
const uploadDocument = buildUploadMiddleware("document", "document");
const uploadAudio = buildUploadMiddleware("audio", "audio");
const uploadChatFile = buildUploadMiddleware("file", "chat");

// Domain: Transport and auth guards.
// Applies CORS restrictions and optional access-code protection.

/**
 * Builds CORS options from configured origin allowlist.
 *
 * @returns {import("cors").CorsOptions} CORS options object.
 */
function buildCorsOptions() {
  const allowedOrigins = new Set(env.corsOrigins);

  return {
    origin: (origin, callback) => {
      if (!origin) {
        return callback(null, true);
      }

      if (allowedOrigins.size === 0 || allowedOrigins.has(origin)) {
        return callback(null, true);
      }

      const error = new Error(`CORS blocked for origin: ${origin}`);
      error.statusCode = 403;
      return callback(error);
    },
    credentials: true,
  };
}

/**
 * Middleware that validates access code headers for protected routes.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @param {import("express").NextFunction} next - Next middleware callback.
 * @returns {void}
 */
function requireAccessCode(req, res, next) {
  if (!env.accessCode) {
    return next();
  }

  // Accept both header names so clients can integrate with less friction.
  const providedCode = req.header("x-access-code") || req.header("access-code");
  if (providedCode !== env.accessCode) {
    return res.status(401).json(buildErrorResponse(req.requestId, "Unauthorized: invalid access code."));
  }

  return next();
}

// Domain: Endpoint handlers.
// Implements request validation, Gemini call orchestration, and API envelopes.

/**
 * Handles POST /generate-text by sending a prompt to Gemini.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @returns {Promise<import("express").Response>} API response.
 */
async function generateText(req, res) {
  const requestId = req.requestId;
  const startedAt = Date.now();

  const prompt = optionalString(req.body?.prompt);
  const modelInput = optionalString(req.body?.model);

  if (!prompt) {
    return res.status(400).json(buildErrorResponse(requestId, 'Field "prompt" is required.'));
  }

  let resolvedModel = null;

  try {
    resolvedModel = resolveGeminiModel(modelInput);

    const rawResponse = await withTimeout(
      geminiClient.models.generateContent({
        model: resolvedModel,
        contents: prompt,
        config: buildGenerationConfig(),
      }),
      GEMINI_TIMEOUT_MS,
      "Gemini generate-text request"
    );

    const output = extractGeminiText(rawResponse);
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: "generate-text",
      statusCode: 200,
      latencyMs,
      model: resolvedModel,
      request: {
        prompt,
        model: modelInput || null,
      },
      response: toLogSafeValue(rawResponse),
      error: null,
    });

    return res.json(
      buildSuccessResponse({
        requestId,
        model: resolvedModel,
        output,
      })
    );
  } catch (error) {
    const normalizedError = normalizeUpstreamError(error);
    const statusCode = normalizedError.statusCode;
    const message = statusCode >= 500 ? `Text generation failed: ${normalizedError.message}` : normalizedError.message;
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: "generate-text",
      statusCode,
      latencyMs,
      model: resolvedModel || modelInput || null,
      request: {
        prompt,
        model: modelInput || null,
      },
      response: null,
      error: { message },
    });

    return res.status(statusCode).json(buildErrorResponse(requestId, message));
  }
}

/**
 * Handles POST /chat with validated message history.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @returns {Promise<import("express").Response>} API response.
 */
async function chat(req, res) {
  const requestId = req.requestId;
  const startedAt = Date.now();

  const file = req.file;
  const modelInput = optionalString(req.body?.model);
  const personalityIdInput = optionalString(req.body?.personalityId);
  const rawMessages = req.body?.messages;

  let resolvedModel = null;
  let resolvedPersonality = null;
  let uploadedGeminiFile = null;

  try {
    if (personalityIdInput) {
      resolvedPersonality = resolvePersonality(personalityIdInput);
    }

    // Validate first, then map once into the exact Gemini payload shape.
    const parsedMessages = parseChatMessagesInput(rawMessages);
    const normalizedMessages = normalizeChatMessages(parsedMessages);
    const chatPayload = buildGeminiChatPayload(normalizedMessages, resolvedPersonality?.systemInstruction || env.systemInstruction);
    resolvedModel = resolveGeminiModel(modelInput);

    if (file) {
      uploadedGeminiFile = await withTimeout(
        uploadFileToGemini(file),
        GEMINI_TIMEOUT_MS,
        "Gemini chat upload request"
      );
    }

    const chatContents = attachUploadedFileToChatContents(chatPayload.contents, uploadedGeminiFile);

    const rawResponse = await withTimeout(
      geminiClient.models.generateContent({
        model: resolvedModel,
        contents: chatContents,
        config: buildGenerationConfig({
          temperature: resolvedPersonality?.generationConfig?.temperature,
          topK: resolvedPersonality?.generationConfig?.topK,
          topP: resolvedPersonality?.generationConfig?.topP,
          maxOutputTokens: resolvedPersonality?.generationConfig?.maxOutputTokens,
          systemInstruction: chatPayload.systemInstruction,
        }),
      }),
      GEMINI_TIMEOUT_MS,
      "Gemini chat request"
    );

    const output = extractGeminiText(rawResponse);
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: "chat",
      statusCode: 200,
      latencyMs,
      model: resolvedModel,
      request: {
        messages: normalizedMessages,
        model: modelInput || null,
        personalityId: personalityIdInput || null,
        file: file
          ? {
              originalName: file.originalname,
              mimeType: file.mimetype,
              size: file.size,
            }
          : null,
      },
      response: toLogSafeValue(rawResponse),
      error: null,
    });

    return res.json(
      buildSuccessResponse({
        requestId,
        model: resolvedModel,
        output,
      })
    );
  } catch (error) {
    const normalizedError = normalizeUpstreamError(error);
    const statusCode = normalizedError.statusCode;
    const message = statusCode >= 500 ? `Chat failed: ${normalizedError.message}` : normalizedError.message;
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: "chat",
      statusCode,
      latencyMs,
      model: resolvedModel || modelInput || null,
      request: {
        messages: Array.isArray(rawMessages) ? rawMessages : typeof rawMessages === "string" ? rawMessages : null,
        model: modelInput || null,
        personalityId: personalityIdInput || null,
        file: file
          ? {
              originalName: file.originalname,
              mimeType: file.mimetype,
              size: file.size,
            }
          : null,
      },
      response: null,
      error: { message },
    });

    return res.status(statusCode).json(buildErrorResponse(requestId, message));
  } finally {
    await removeFileIfExists(file?.path);
    await safeDeleteGeminiFile(uploadedGeminiFile?.name, requestId);
  }
}

/**
 * Shared handler for image, document, and audio generation endpoints.
 *
 * @param {import("express").Request} req - Incoming request with uploaded file.
 * @param {import("express").Response} res - Outgoing response.
 * @param {{endpointName: string, requiresPrompt: boolean, defaultPrompt: string}} options - Route-specific behavior.
 * @returns {Promise<import("express").Response>} API response.
 */
async function handleFileEndpoint(req, res, options) {
  const requestId = req.requestId;
  const startedAt = Date.now();

  const { endpointName, requiresPrompt, defaultPrompt } = options;

  const file = req.file;
  const prompt = optionalString(req.body?.prompt);
  const modelInput = optionalString(req.body?.model);

  if (!file) {
    return res.status(400).json(buildErrorResponse(requestId, "Upload file is required."));
  }

  if (requiresPrompt && !prompt) {
    // Clean up immediately when request validation fails after upload.
    await removeFileIfExists(file.path);
    return res.status(400).json(buildErrorResponse(requestId, 'Field "prompt" is required.'));
  }

  let resolvedModel = null;
  let uploadedGeminiFile = null;

  try {
    resolvedModel = resolveGeminiModel(modelInput);

    // Upload local temp file first, then pass Gemini a URI part instead of base64.
    const finalPrompt = prompt || defaultPrompt;
    uploadedGeminiFile = await withTimeout(
      uploadFileToGemini(file),
      GEMINI_TIMEOUT_MS,
      `Gemini ${endpointName} upload request`
    );

    const rawResponse = await withTimeout(
      geminiClient.models.generateContent({
        model: resolvedModel,
        contents: [
          { text: finalPrompt },
          createPartFromUri(uploadedGeminiFile.uri, uploadedGeminiFile.mimeType),
        ],
        config: buildGenerationConfig(),
      }),
      GEMINI_TIMEOUT_MS,
      `Gemini ${endpointName} request`
    );

    const output = extractGeminiText(rawResponse);
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: endpointName,
      statusCode: 200,
      latencyMs,
      model: resolvedModel,
      request: {
        prompt: prompt || null,
        model: modelInput || null,
        file: {
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
        },
      },
      response: toLogSafeValue(rawResponse),
      error: null,
    });

    return res.json(
      buildSuccessResponse({
        requestId,
        model: resolvedModel,
        output,
      })
    );
  } catch (error) {
    const normalizedError = normalizeUpstreamError(error);
    const statusCode = normalizedError.statusCode;
    const message = statusCode >= 500 ? `${endpointName} failed: ${normalizedError.message}` : normalizedError.message;
    const latencyMs = Date.now() - startedAt;

    await safeWriteRequestLog({
      timestamp: new Date().toISOString(),
      requestId,
      endpoint: endpointName,
      statusCode,
      latencyMs,
      model: resolvedModel || modelInput || null,
      request: {
        prompt: prompt || null,
        model: modelInput || null,
        file: {
          originalName: file.originalname,
          mimeType: file.mimetype,
          size: file.size,
        },
      },
      response: null,
      error: { message },
    });

    return res.status(statusCode).json(buildErrorResponse(requestId, message));
  } finally {
    // Always remove temporary uploads, even on upstream or network errors.
    await removeFileIfExists(file.path);
    // Also clean up uploaded Gemini files so one-off requests do not accumulate storage.
    await safeDeleteGeminiFile(uploadedGeminiFile?.name, requestId);
  }
}

/**
 * Handles POST /generate-from-image.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @returns {Promise<import("express").Response>} API response.
 */
async function generateFromImage(req, res) {
  return handleFileEndpoint(req, res, {
    endpointName: "generate-from-image",
    requiresPrompt: true,
    defaultPrompt: "Analyze this image and provide a concise answer.",
  });
}

/**
 * Handles POST /generate-from-document.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @returns {Promise<import("express").Response>} API response.
 */
async function generateFromDocument(req, res) {
  return handleFileEndpoint(req, res, {
    endpointName: "generate-from-document",
    requiresPrompt: false,
    defaultPrompt: "Analyze this document and summarize key points.",
  });
}

/**
 * Handles POST /generate-from-audio.
 *
 * @param {import("express").Request} req - Incoming request.
 * @param {import("express").Response} res - Outgoing response.
 * @returns {Promise<import("express").Response>} API response.
 */
async function generateFromAudio(req, res) {
  return handleFileEndpoint(req, res, {
    endpointName: "generate-from-audio",
    requiresPrompt: false,
    defaultPrompt: "Transcribe and analyze this audio content.",
  });
}

// Domain: Core utilities and configuration.
// Lowest-level helpers and env parsing used by higher domains above.

/**
 * Converts a comma-separated string into a trimmed array.
 *
 * @param {unknown} value - Raw value to parse, usually from env.
 * @param {string[]} fallback - Returned when parsed values are empty.
 * @returns {string[]} Parsed values or fallback.
 */
function splitCsv(value, fallback = []) {
  const source = typeof value === "string" ? value : "";
  const parsed = source
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  return parsed.length > 0 ? parsed : fallback;
}

/**
 * Parses a finite number from a raw input value.
 *
 * @param {unknown} value - Raw value to parse.
 * @param {number} fallback - Returned when parsing fails.
 * @returns {number} Parsed number or fallback.
 */
function parseNumber(value, fallback) {
  if (typeof value !== "string" || value.trim() === "") {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

/**
 * Normalizes a value into a trimmed string.
 *
 * @param {unknown} value - Raw value to normalize.
 * @returns {string} Trimmed string or empty string.
 */
function optionalString(value) {
  return typeof value === "string" ? value.trim() : "";
}

/**
 * Creates an Error object with an HTTP status code attached.
 *
 * @param {number} statusCode - HTTP status to use in responses.
 * @param {string} message - Human-readable error message.
 * @returns {Error & {statusCode: number}} Error instance with statusCode.
 */
function createHttpError(statusCode, message) {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
}

/**
 * Resolves upstream errors into a valid HTTP status code and readable message.
 *
 * @param {any} error - Unknown error object from SDK/network/runtime.
 * @returns {{statusCode: number, message: string}} Normalized HTTP error fields.
 */
function normalizeUpstreamError(error) {
  let parsedErrorPayload = null;

  if (typeof error?.message === "string") {
    try {
      parsedErrorPayload = JSON.parse(error.message);
    } catch (_parseError) {
      parsedErrorPayload = null;
    }
  }

  const statusCandidates = [
    error?.statusCode,
    error?.status,
    error?.code,
    error?.response?.status,
    error?.response?.statusCode,
    parsedErrorPayload?.error?.code,
    parsedErrorPayload?.code,
  ];

  let statusCode = 500;

  for (const candidate of statusCandidates) {
    const numericCandidate = Number(candidate);
    if (Number.isInteger(numericCandidate) && numericCandidate >= 400 && numericCandidate <= 599) {
      statusCode = numericCandidate;
      break;
    }
  }

  const message =
    optionalString(parsedErrorPayload?.error?.message) ||
    optionalString(parsedErrorPayload?.message) ||
    optionalString(error?.message) ||
    "Unexpected server error.";

  return {
    statusCode,
    message,
  };
}

/**
 * Throws when a required condition is not met.
 *
 * @param {boolean} condition - Condition that must be truthy.
 * @param {string} message - Error message for failed conditions.
 * @throws {Error} When condition is false.
 */
function assertCondition(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

/**
 * Reads, normalizes, and validates runtime configuration from environment variables.
 *
 * @returns {{
 * nodeEnv: string,
 * port: number,
 * corsOrigins: string[],
 * geminiApiKey: string,
 * geminiDefaultModel: string,
 * geminiAllowedModels: string[],
 * geminiTemperature: number,
 * geminiTopK: number,
 * geminiTopP: number,
 * geminiMaxTokens: number,
 * systemInstruction: string,
 * maxUploadSizeBytes: number,
 * allowedUploadExtensions: string[],
 * accessCode: string
 * }} Runtime configuration object.
 * @throws {Error} When required env values are missing or invalid.
 */
function buildEnv() {
  const allowedExtensionsFallback = ["pdf", "txt", "mp3", "wav", "png", "bmp", "jpg", "jpeg", "webp"];
  const maxUploadSizeMb = parseNumber(process.env.MAX_UPLOAD_SIZE_MB, DEFAULT_MAX_UPLOAD_SIZE_MB);

  // Normalize env values into the types the app actually uses.
  const env = {
    nodeEnv: optionalString(process.env.NODE_ENV) || "development",
    port: parseNumber(process.env.PORT, 3000),
    corsOrigins: splitCsv(process.env.CORS_ORIGIN, []),
    geminiApiKey: optionalString(process.env.GEMINI_API_KEY),
    geminiDefaultModel: optionalString(process.env.GEMINI_DEFAULT_MODEL) || "gemini-2.5-flash-lite",
    geminiAllowedModels: splitCsv(process.env.GEMINI_ALLOWED_MODELS, ["gemini-2.5-flash-lite"]),
    geminiTemperature: parseNumber(process.env.GEMINI_TEMPERATURE, 0.7),
    geminiTopK: parseNumber(process.env.GEMINI_TOP_K, 40),
    geminiTopP: parseNumber(process.env.GEMINI_TOP_P, 0.9),
    geminiMaxTokens: parseNumber(process.env.GEMINI_MAX_TOKENS, 8192),
    systemInstruction: optionalString(process.env.SYSTEM_INSTRUCTION),
    maxUploadSizeBytes: Math.floor(maxUploadSizeMb * 1024 * 1024),
    allowedUploadExtensions: splitCsv(process.env.ALLOWED_UPLOAD_EXTENSIONS, allowedExtensionsFallback).map((item) =>
      item.toLowerCase()
    ),
    accessCode: optionalString(process.env.ACCESS_CODE),
  };

  // Fail fast on invalid config so the app does not run in a broken state.
  assertCondition(Boolean(env.geminiApiKey), "Missing GEMINI_API_KEY in .env.");
  assertCondition(Number.isInteger(env.port) && env.port > 0 && env.port <= 65535, "PORT must be a valid number between 1 and 65535.");
  assertCondition(env.geminiAllowedModels.length > 0, "GEMINI_ALLOWED_MODELS must contain at least one model.");
  assertCondition(env.geminiAllowedModels.includes(env.geminiDefaultModel), "GEMINI_DEFAULT_MODEL must exist in GEMINI_ALLOWED_MODELS.");
  // Numeric guardrails prevent unstable model behavior from invalid values.
  assertCondition(env.geminiTemperature >= 0 && env.geminiTemperature <= 2, "GEMINI_TEMPERATURE must be between 0 and 2.");
  assertCondition(env.geminiTopK > 0, "GEMINI_TOP_K must be greater than 0.");
  assertCondition(env.geminiTopP > 0 && env.geminiTopP <= 1, "GEMINI_TOP_P must be greater than 0 and less than or equal to 1.");
  assertCondition(env.geminiMaxTokens > 0, "GEMINI_MAX_TOKENS must be greater than 0.");
  assertCondition(maxUploadSizeMb > 0, "MAX_UPLOAD_SIZE_MB must be greater than 0.");

  return env;
}

bootstrap().catch((error) => {
  console.error("Failed to start API V2 server:", error.message);
  process.exit(1);
});
