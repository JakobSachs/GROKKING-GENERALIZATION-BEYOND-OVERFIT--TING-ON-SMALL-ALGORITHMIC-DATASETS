import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { initialContext, SYSTEM_PROMPT, WEIGHT_QUESTION, WEIGHT_TOOL } from "../blind_protocol.mjs";
import { numericalTool, run } from "../run_blind_sol.mjs";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const ARCHITECTURE = JSON.parse(fs.readFileSync(path.join(ROOT, "blind_architecture.json"), "utf8"));
const digest = (file) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "blind-harness-test-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
}

function fakeSdk(directory, { fail = false } = {}) {
  const sdk = path.join(directory, "sdk.mjs");
  fs.writeFileSync(sdk, `export class ModelRuntime {
    static async create() { return new ModelRuntime(); }
    getModel(provider,id) { return {provider,id}; }
    async getAuth() { return {}; }
    async completeSimple(model,context,options) {
      if (options.maxRetries !== 0 || options.cacheRetention !== 'none') throw new Error('Unsafe request policy');
      options.onPayload({messages: context.messages, tools: context.tools ?? []});
      this.calls = (this.calls ?? 0) + 1;
      ${fail ? "return {role:'assistant',stopReason:'error',content:[],errorMessage:'fixture'};" : `
      if (this.calls === 1) return {role:'assistant',stopReason:'stop',content:[{type:'text',text:'Architecture-only prior fixture.'}]};
      if (this.calls === 2) return {role:'assistant',stopReason:'toolUse',content:[{type:'toolCall',id:'call-fixture',name:'inspect_weights',arguments:{operation:'catalog'}}]};
      return {role:'assistant',stopReason:'stop',content:[{type:'text',text:'Uncertain task; final weight assessment fixture.'}]};`}
    }
  }`);
  return sdk;
}

function argumentsFor(directory, sdk) {
  const architecture = path.join(directory, "architecture.json");
  fs.writeFileSync(architecture, JSON.stringify(ARCHITECTURE));
  const specimen = path.join(directory, "specimen.npz");
  fs.writeFileSync(specimen, "fixture bytes");
  const inspector = path.join(directory, "inspector.sh");
  fs.writeFileSync(inspector, "printf '{\"tensors\":[]}'\n");
  const receipt = path.join(directory, "receipt.json");
  fs.writeFileSync(receipt, JSON.stringify({ specimen_sha256: digest(specimen), architecture_sha256: digest(architecture) }));
  return { sdk, python: "/bin/sh", specimen, architecture, inspector, receipt, output: path.join(directory, "result") };
}

test("initial context is architecture-only, with no tools, source or task labels", () => {
  const context = initialContext(ARCHITECTURE);
  assert.equal(context.messages.length, 1);
  assert.equal(context.tools, undefined);
  const text = JSON.stringify({ context, SYSTEM_PROMPT, WEIGHT_QUESTION, WEIGHT_TOOL }).toLowerCase();
  for (const forbidden of ["grokking", "modulo", "modular", "a+b", "separator", "trained on", "jakob", "/home/", "accuracy", "3844"]) {
    assert.equal(text.includes(forbidden), false, forbidden);
  }
  assert.match(context.messages[0].content, /Architecture:/);
  assert.deepEqual(WEIGHT_TOOL.parameters.required, ["operation"]);
});

test("numerical process failure does not expose stderr or host paths", (t) => {
  const directory = fixture(t);
  const inspector = path.join(directory, "failure.sh");
  fs.writeFileSync(inspector, `echo '${directory}/private-file' >&2\nexit 1\n`);
  const result = numericalTool({ python: "/bin/sh", inspector, specimen: "/private/specimen" }, {}, directory);
  assert.equal(result.isError, true);
  assert.equal(result.text.includes(directory), false);
  assert.equal(result.text.includes("private-file"), false);
});

test("fresh explicit SDK context records prior, numerical exchange and final answer", async (t) => {
  const directory = fixture(t);
  const args = argumentsFor(directory, fakeSdk(directory));
  await run(args);
  const info = JSON.parse(fs.readFileSync(path.join(args.output, "run.json"), "utf8"));
  assert.equal(info.status, "completed");
  assert.equal(info.requestCount, 3);
  assert.equal(info.toolCalls, 1);
  assert.equal(info.model, "gpt-6.1-sol");
  const events = fs.readFileSync(path.join(args.output, "transcript.jsonl"), "utf8").trim().split("\n").map(JSON.parse);
  const payloads = events.filter((event) => event.type === "provider_payload");
  assert.equal(payloads[0].payload.messages.length, 1);
  assert.equal(payloads[0].payload.tools.length, 0);
  for (const payload of payloads) {
    assert.equal(JSON.stringify(payload).includes(directory), false);
    assert.equal(JSON.stringify(payload).includes("fixture bytes"), false);
  }
  assert.match(fs.readFileSync(path.join(args.output, "architecture-only.md"), "utf8"), /prior fixture/);
  assert.match(fs.readFileSync(path.join(args.output, "assessment.md"), "utf8"), /weight assessment/);
  const originalRun = fs.readFileSync(path.join(args.output, "run.json"));
  const originalPrior = fs.readFileSync(path.join(args.output, "architecture-only.md"));
  await assert.rejects(run(args), /EEXIST/);
  assert.deepEqual(fs.readFileSync(path.join(args.output, "run.json")), originalRun);
  assert.deepEqual(fs.readFileSync(path.join(args.output, "architecture-only.md")), originalPrior);
});

test("altered specimen is rejected before a model request", async (t) => {
  const directory = fixture(t);
  const args = argumentsFor(directory, fakeSdk(directory));
  fs.appendFileSync(args.specimen, "tampered");
  await assert.rejects(run(args), /differs from frozen export/);
  assert.equal(fs.existsSync(args.output), false);
});

test("extra architecture fields cannot leak training metadata", async (t) => {
  const directory = fixture(t);
  const args = argumentsFor(directory, fakeSdk(directory));
  fs.writeFileSync(args.architecture, JSON.stringify({ ...ARCHITECTURE, task: "private answer" }));
  await assert.rejects(run(args), /only the reviewed/);
  assert.equal(fs.existsSync(args.output), false);
});

test("failed request is preserved without automatic retry", async (t) => {
  const directory = fixture(t);
  const args = argumentsFor(directory, fakeSdk(directory, { fail: true }));
  await assert.rejects(run(args), /did not complete normally/);
  const info = JSON.parse(fs.readFileSync(path.join(args.output, "run.json"), "utf8"));
  assert.equal(info.status, "failed");
  assert.equal(info.requestCount, 1);
  assert.equal(info.toolCalls, 0);
  assert.equal(fs.existsSync(path.join(args.output, "assessment.md")), false);
});
