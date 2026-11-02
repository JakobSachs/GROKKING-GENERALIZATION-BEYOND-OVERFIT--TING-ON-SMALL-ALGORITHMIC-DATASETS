import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { initialContext, messageText, WEIGHT_QUESTION, WEIGHT_TOOL } from "./blind_protocol.mjs";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const MAX_TOOL_CALLS = 20;
const MAX_TOOL_OUTPUT = 12000;
const WALL_TIME_MS = 20 * 60 * 1000;

function sha256(filename) {
  return createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
}

function parseArgs(args) {
  const result = {};
  const allowed = new Set(["sdk", "python", "specimen", "architecture", "inspector", "receipt", "output"]);
  for (let i = 0; i < args.length; i += 2) {
    const name = args[i]?.replace(/^--/, "");
    if (!allowed.has(name) || !args[i + 1] || result[name]) throw new Error("Invalid CLI arguments");
    result[name] = path.resolve(args[i + 1]);
  }
  for (const name of allowed) if (!result[name]) throw new Error(`Missing --${name}`);
  return result;
}

export function numericalTool(args, request, workingDirectory) {
  const processResult = spawnSync(args.python, [args.inspector, "--specimen", args.specimen], {
    input: JSON.stringify(request),
    encoding: "utf8",
    cwd: workingDirectory,
    timeout: 30000,
    maxBuffer: 1024 * 1024,
    env: {
      PATH: "/usr/bin:/bin",
      OPENBLAS_NUM_THREADS: "2",
      OMP_NUM_THREADS: "2",
      PYTHONNOUSERSITE: "1",
      PYTHONDONTWRITEBYTECODE: "1",
    },
  });
  if (processResult.error || processResult.signal || !processResult.stdout || processResult.stdout.length > MAX_TOOL_OUTPUT) {
    return { text: JSON.stringify({ error: "Numerical inspection failed or exceeded its resource limit." }), isError: true };
  }
  try {
    const result = JSON.parse(processResult.stdout);
    return { text: JSON.stringify(result), isError: processResult.status !== 0 || Boolean(result.error) };
  } catch {
    return { text: JSON.stringify({ error: "Numerical inspection returned an invalid response." }), isError: true };
  }
}

export async function run(args) {
  const architecture = JSON.parse(fs.readFileSync(args.architecture, "utf8"));
  const expected = JSON.parse(fs.readFileSync(path.join(ROOT, "blind_architecture.json"), "utf8"));
  assert.deepEqual(architecture, expected, "Architecture must contain only the reviewed, fixed high-level information");
  const receipt = JSON.parse(fs.readFileSync(args.receipt, "utf8"));
  assert.equal(sha256(args.specimen), receipt.specimen_sha256, "Specimen differs from frozen export");
  assert.equal(sha256(args.architecture), receipt.architecture_sha256, "Architecture differs from frozen export");
  fs.mkdirSync(path.dirname(args.output), { recursive: true });
  fs.mkdirSync(args.output);
  const neutralDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "tensor-inspection-"));
  const runPath = path.join(args.output, "run.json");
  const transcriptPath = path.join(args.output, "transcript.jsonl");
  const runInfo = {
    status: "prepared",
    provider: "openai",
    model: "gpt-6.1-sol",
    sessionId: randomUUID(),
    specimen_sha256: sha256(args.specimen),
    architecture_sha256: sha256(args.architecture),
    protocol_sha256: sha256(path.join(ROOT, "blind_protocol.mjs")),
    inspector_sha256: sha256(args.inspector),
    runner_sha256: sha256(fileURLToPath(import.meta.url)),
    requestCount: 0,
    toolCalls: 0,
    maxRetries: 0,
    maxToolCalls: MAX_TOOL_CALLS,
  };
  const saveInfo = () => fs.writeFileSync(runPath, `${JSON.stringify(runInfo, null, 2)}\n`);
  const record = (event) => fs.appendFileSync(transcriptPath, `${JSON.stringify(event)}\n`);
  saveInfo();
  try {
    const { ModelRuntime } = await import(pathToFileURL(args.sdk).href);
    const runtime = await ModelRuntime.create({
      authPath: path.join(os.homedir(), ".pi/agent/auth.json"),
      modelsPath: null,
      refreshOnCreate: false,
    });
    const model = runtime.getModel("openai", "gpt-6.1-sol");
    if (!model) throw new Error("Requested model unavailable in local catalog");
    if (!(await runtime.getAuth("openai"))) throw new Error("Configured subscription unavailable");
    const deadline = Date.now() + WALL_TIME_MS;
    const context = initialContext(architecture);
    record({ type: "initial_context", context });

    async function request() {
      const remaining = deadline - Date.now();
      if (remaining <= 0) throw new Error("Blind assessment exceeded wall-time limit");
      runInfo.requestCount += 1;
      runInfo.status = "requesting";
      saveInfo();
      record({ type: "outbound_context", request: runInfo.requestCount, context });
      const response = await runtime.completeSimple(model, context, {
        reasoning: "high",
        maxTokens: 8192,
        maxRetries: 0,
        cacheRetention: "none",
        sessionId: runInfo.sessionId,
        signal: AbortSignal.timeout(remaining),
        onPayload: (payload) => record({ type: "provider_payload", request: runInfo.requestCount, payload }),
      });
      record({ type: "assistant_message", request: runInfo.requestCount, response });
      context.messages.push(response);
      if (!["stop", "toolUse"].includes(response.stopReason)) {
        throw new Error(`Model request did not complete normally (${response.stopReason})`);
      }
      return response;
    }

    const prior = await request();
    assert.equal(prior.stopReason, "stop", "Architecture-only phase cannot use tools");
    const priorText = messageText(prior);
    if (!priorText.trim()) throw new Error("Architecture-only assessment was empty");
    fs.writeFileSync(path.join(args.output, "architecture-only.md"), `${priorText}\n`, { flag: "wx" });
    context.tools = [WEIGHT_TOOL];
    context.messages.push({ role: "user", content: WEIGHT_QUESTION, timestamp: Date.now() });

    for (let turn = 0; turn <= MAX_TOOL_CALLS + 1; turn += 1) {
      const response = await request();
      const calls = response.content.filter((block) => block.type === "toolCall");
      if (response.stopReason === "stop" && calls.length === 0) {
        const answer = messageText(response);
        if (!answer.trim()) throw new Error("Final assessment was empty");
        fs.writeFileSync(path.join(args.output, "assessment.md"), `${answer}\n`, { flag: "wx" });
        runInfo.status = "completed";
        saveInfo();
        console.log(`Blind assessment saved. Requests: ${runInfo.requestCount}; numerical calls: ${runInfo.toolCalls}.`);
        return;
      }
      if (response.stopReason !== "toolUse" || calls.length === 0) throw new Error("Malformed tool response");
      for (const call of calls) {
        let result;
        if (call.name !== WEIGHT_TOOL.name) {
          result = { text: JSON.stringify({ error: "Unknown tool." }), isError: true };
        } else if (runInfo.toolCalls >= MAX_TOOL_CALLS) {
          result = { text: JSON.stringify({ error: "Tool budget exhausted. Give your final assessment." }), isError: true };
        } else {
          runInfo.toolCalls += 1;
          result = numericalTool(args, call.arguments, neutralDirectory);
        }
        const message = {
          role: "toolResult",
          toolCallId: call.id,
          toolName: call.name,
          content: [{ type: "text", text: result.text }],
          isError: result.isError,
          timestamp: Date.now(),
        };
        record({ type: "numerical_result", message });
        context.messages.push(message);
      }
      saveInfo();
      if (runInfo.toolCalls >= MAX_TOOL_CALLS) {
        context.tools = [];
        context.messages.push({ role: "user", content: "The numerical inspection budget is exhausted. Give your final assessment from the evidence already obtained.", timestamp: Date.now() });
      }
    }
    throw new Error("Model did not finish within the bounded interaction budget");
  } catch (error) {
    runInfo.status = "failed";
    runInfo.errorType = error instanceof Error ? error.name : "UnknownError";
    saveInfo();
    throw error;
  } finally {
    fs.rmSync(neutralDirectory, { recursive: true, force: true });
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run(parseArgs(process.argv.slice(2))).catch((error) => {
    // SDK details stay out of model-visible tools and terminal logs.
    console.error(`Blind assessment stopped (${error instanceof Error ? error.name : "UnknownError"}); inspect the private run record.`);
    process.exitCode = 1;
  });
}
