import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

function loadConversationStore() {
  const items = new Map();
  const storage = {
    async getItem(key) { return structuredClone(items.get(key) ?? null); },
    async setItem(key, value) { items.set(key, structuredClone(value)); },
    async removeItem(key) { items.delete(key); },
  };
  const source = readFileSync(new URL("../src/store/image-conversations.ts", import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports,
    require(name) {
      assert.equal(name, "localforage");
      return { createInstance: () => storage };
    },
  });
  return exports;
}

function conversation() {
  return {
    id: "animation", title: "Animated cat", model: "gpt-image-2.6-flare",
    createdAt: "2026-11-11T12:00:00Z", updatedAt: "2026-11-11T12:00:00Z",
    turns: [{
      id: "turn", prompt: "A jumping cat", model: "gpt-image-2.6-flare", mode: "generate",
      referenceImages: [], maskImages: [], count: 1, size: "1024x1024", ratio: "1:1", tier: "1k",
      quality: "auto", outputFormat: "gif", frameCount: 12, frameDurationMs: 150,
      images: [], createdAt: "2026-11-11T12:00:00Z", status: "queued",
    }],
  };
}

function assertAnimation(turn) {
  assert.equal(turn.outputFormat, "gif");
  assert.equal(turn.frameCount, 12);
  assert.equal(turn.frameDurationMs, 150);
}

test("saving and reading a conversation preserves GIF generation parameters", async () => {
  const store = loadConversationStore();
  await store.saveImageConversation(conversation());
  const [saved] = await store.listImageConversations();
  assertAnimation(saved.turns[0]);
});

test("bulk saves and renames preserve animation parameters alongside legacy turns", async () => {
  const store = loadConversationStore();
  const value = conversation();
  const legacy = { ...value.turns[0], id: "legacy" };
  delete legacy.outputFormat;
  delete legacy.frameCount;
  delete legacy.frameDurationMs;
  value.turns.push(legacy);
  await store.saveImageConversations([value]);
  await store.renameImageConversation(value.id, "Renamed animation");
  const [saved] = await store.listImageConversations();
  assert.equal(saved.title, "Renamed animation");
  assertAnimation(saved.turns[0]);
  assert.equal(saved.turns[1].outputFormat, "png");
  assert.equal(saved.turns[1].frameCount, undefined);
  assert.equal(saved.turns[1].frameDurationMs, undefined);
});
