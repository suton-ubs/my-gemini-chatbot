# My Gemini Chatbot

Simple API implementation for chatbot based on Gemini AI.

## Tech Stack
- Node.js + Express
- Gemini SDK: `@google/genai`
- Multer for uploads

## Endpoints
Public:
- `GET /health`
- `GET /chat-meta`

Protected (when `GEMINI_API_KEY` is set):
- `POST /generate-text`
- `POST /chat`
- `POST /generate-from-image`
- `POST /generate-from-document`
- `POST /generate-from-audio`

## Environment Variables
This implementation intentionally uses only these keys from `.env`:

- `NODE_ENV`
- `PORT`
- `CORS_ORIGIN`
- `GEMINI_API_KEY` (optional; when empty, the UI asks for it and sends it per request)
- `GEMINI_DEFAULT_MODEL`
- `GEMINI_ALLOWED_MODELS`
- `GEMINI_TEMPERATURE`
- `GEMINI_TOP_K`
- `GEMINI_TOP_P`
- `GEMINI_MAX_TOKENS`
- `SYSTEM_INSTRUCTION`
- `ALLOWED_UPLOAD_EXTENSIONS`
- `MAX_UPLOAD_SIZE_MB`
- `ACCESS_CODE`

Notes:
- `GEMINI_DEFAULT_MODEL` must exist in `GEMINI_ALLOWED_MODELS`.
- `MAX_UPLOAD_SIZE_MB` must be greater than `0`.
- `ALLOWED_UPLOAD_EXTENSIONS` is global allowlist and is intersected with route-specific rules for image/document/audio routes.

## Run Locally
1. Go to folder:

```bash
cd my-gemini-chatbot
```

2. Install dependencies:

```bash
npm install
```

3. Start in dev mode:

```bash
npm run dev
```

Or run normally:

```bash
npm run start
```

## Authentication
When `GEMINI_API_KEY` exists in `.env`, protected endpoints require one of these headers:

- `x-access-code: <ACCESS_CODE>`
- `access-code: <ACCESS_CODE>`

If `GEMINI_API_KEY` is empty, the app switches to bring-your-own-key mode: the UI prompts for a Gemini API key, and `ACCESS_CODE` is disabled.

Open endpoints:
- `GET /health`
- `GET /chat-meta`

## Request ID Tracing
- Every request gets `requestId` and the server also returns `x-request-id` response header.
- Clients may send `x-request-id`; when missing, server generates one.
- Success and error API envelopes include `requestId`.

## Chat Contract (Gemini Native)
`POST /chat` requires:

- `messages`: array (or JSON string when using multipart form)
- `personalityId`: optional string (must match an `id` in `personalities.json`)
- `file`: optional upload file (multipart/form-data)
- each message has:
  - `role`: `system` | `user` | `model`
  - `text` (or `content`/`parts`)

Accepted content types:
- `application/json` for normal chat requests
- `multipart/form-data` when sending `file`

When using `multipart/form-data`:
- send `messages` as JSON string
- `file` extension must be listed in `.env` `ALLOWED_UPLOAD_EXTENSIONS`
- `file` size must be <= `.env` `MAX_UPLOAD_SIZE_MB`
- uploaded `file` is attached to latest `user` message

Rules:
- at least one `user` message
- last message must be `user`

System instruction precedence:
- if payload has one or more `system` messages, that content is used
- otherwise fallback to selected personality `config.system_instruction` when `personalityId` is provided
- if `personalityId` is blank, fallback to `.env` `SYSTEM_INSTRUCTION`

Chat generation settings precedence (`temperature`, `topK`, `topP`, `maxOutputTokens`):
- if `personalityId` is provided and valid, values come from that personality config
- if `personalityId` is blank, values come from `.env`

Personality validation:
- `personalityId` must exist in `my-gemini-chatbot/personalities.json`
- unknown `personalityId` returns HTTP `400`

## File Endpoint Contract
`POST /generate-from-image`
- multipart field: `image`
- `prompt` is required

`POST /generate-from-document`
- multipart field: `document`
- `prompt` is optional (server uses fallback prompt when omitted)

`POST /generate-from-audio`
- multipart field: `audio`
- `prompt` is optional (server uses fallback prompt when omitted)

Route extension policy:
- chat file: any extension from `ALLOWED_UPLOAD_EXTENSIONS`
- image route: `png`, `bmp`, `jpg`, `jpeg`, `webp` intersected with `ALLOWED_UPLOAD_EXTENSIONS`
- document route: `pdf`, `txt` intersected with `ALLOWED_UPLOAD_EXTENSIONS`
- audio route: `mp3`, `wav` intersected with `ALLOWED_UPLOAD_EXTENSIONS`

Example:

```json
{
  "model": "gemini-2.5-flash-lite",
  "personalityId": "personalities-01",
  "messages": [
    { "role": "system", "text": "Answer briefly." },
    { "role": "user", "text": "Give 3 API logging best practices." }
  ]
}
```

Multipart example (`curl`):

```bash
curl -X POST http://localhost:3000/chat \
  -H "x-access-code: six.seven" \
  -F "model=gemini-2.5-flash-lite" \
  -F "personalityId=personalities-03" \
  -F "messages=[{\"role\":\"user\",\"text\":\"Analyze this file\"}]" \
  -F "file=@C:/path/to/your-file.pdf"
```

## Personality Catalog
Personalities are loaded from `my-gemini-chatbot/personalities.json` at server startup.

Each personality contains:
- `id`
- `title`
- `short_description`
- `config.gemini_temperature`
- `config.gemini_top_k`
- `config.gemini_top_p`
- `config.gemini_max_tokens`
- `config.system_instruction`

If `personalities.json` does not exist, chat still works; in that case, only blank `personalityId` is valid and `.env` settings are used.

## Response Shape
Standard success response (`/generate-text`, `/chat`, file endpoints):

```json
{
  "status": "success",
  "requestId": "...",
  "model": "gemini-2.5-flash-lite",
  "output": "..."
}
```

Standard error response:

```json
{
  "status": "error",
  "requestId": "...",
  "error": "..."
}
```

`GET /health` response includes:
- `status`
- `requestId`
- `uptimeSeconds`
- `timestamp`

`GET /chat-meta` response includes:
- `status`
- `requestId`
- `defaultModel`
- `allowedModels`
- `allowedUploadExtensions` (for chat uploads)
- `personalities` (`id`, `title`, `shortDescription`)

## UI (public Folder)
`public/` is a tiny learning/demo UI served as static files.

- URL: `http://localhost:3000/`
- Purpose: quick manual test for chat
- Not required for API usage (Postman/curl works too)

## Logs and Uploads
At runtime, app automatically creates:

- `logs/`
- `uploads/`

Logs are JSON per request with key metadata:
- requestId
- endpoint
- statusCode
- latencyMs
- model

Request log payload also stores request snapshot and error/response payload in JSON-safe form.

For `/chat`, request logs include `personalityId` under `request` when provided.

Uploaded files are deleted after request processing.

## Learning Walkthrough (index.js)
Key areas to read in order:
1. Env parsing and fail-fast validation
2. Request id middleware
3. Access code middleware
4. Chat normalization and Gemini mapping
5. Upload endpoint handler and cleanup
6. Centralized error handling
7. Graceful shutdown

## License
This project is licensed under the MIT License.
