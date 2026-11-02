export const SYSTEM_PROMPT = `You are analyzing an unfamiliar trained neural network. You have no prior conversation or task description. Use only the architecture and numerical weight evidence supplied here. Distinguish architectural priors from evidence in learned weights. Do not assume the task is identifiable. Do not execute the network or any layer on real, random, synthetic, or imagined input vectors; weight-to-weight matrix analysis is allowed. There is no internet or access to source, data, labels, examples, or model outputs. State uncertainty honestly.`;

export const INITIAL_QUESTION = `Before inspecting any weights, what broad task families are plausible from this architecture alone? Give a brief, explicitly tentative prior assessment, not a confident identification. You will be given access to the numerical tensors afterward.`;

export const WEIGHT_QUESTION = `You can now inspect the complete learned tensors with the numerical tool. What do you think these weights solve? Look for numerical evidence that discriminates tasks, rather than relying only on dimensions. You have at most 20 tool calls. Finish with ranked hypotheses, calibrated confidence, concrete weight-based evidence, and what remains unknowable. Do not request input examples or run forward passes.`;

const tensorSelection = {
  type: "object",
  properties: {
    tensor: { type: "string", description: "An exact tensor name from the catalog." },
    row_start: { type: "integer", minimum: 0 },
    rows: { type: "integer", minimum: 1, maximum: 512 },
    col_start: { type: "integer", minimum: 0 },
    cols: { type: "integer", minimum: 1, maximum: 512 },
    transpose: { type: "boolean" },
  },
  required: ["tensor"],
  additionalProperties: false,
};

export const WEIGHT_TOOL = {
  name: "inspect_weights",
  description: "Read learned tensor values or apply bounded generic matrix analysis. Start with operation=catalog for tensor names, shapes and statistics. For other operations select left; optional right computes left @ right after selection/transposition. Select the returned result using output_row_start/output_rows and output_col_start/output_cols. No model execution, input probes, source files or other data are available.",
  parameters: {
    type: "object",
    properties: {
      operation: {
        type: "string",
        enum: ["catalog", "values", "stats", "row_norms", "column_norms", "cosine_rows", "cosine_columns", "singular_values", "left_singular_vectors", "right_singular_vectors", "fft_rows_abs", "fft_columns_abs"],
      },
      left: tensorSelection,
      right: tensorSelection,
      output_row_start: { type: "integer", minimum: 0 },
      output_rows: { type: "integer", minimum: 1, maximum: 384 },
      output_col_start: { type: "integer", minimum: 0 },
      output_cols: { type: "integer", minimum: 1, maximum: 384 },
    },
    required: ["operation"],
    additionalProperties: false,
  },
};

export function initialContext(architecture) {
  return {
    systemPrompt: SYSTEM_PROMPT,
    messages: [{ role: "user", content: `${INITIAL_QUESTION}\n\nArchitecture:\n${JSON.stringify(architecture, null, 2)}`, timestamp: Date.now() }],
  };
}

export function messageText(message) {
  return message.content.filter((block) => block.type === "text").map((block) => block.text).join("\n");
}
