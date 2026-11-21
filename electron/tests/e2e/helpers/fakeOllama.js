// SPDX-License-Identifier: MIT
/**
 * A stand-in for the local model server (Ollama), for model-query.spec.js: the question path is
 * tested through the real window and the real API with the model mocked at its HTTP boundary, so
 * CI needs no Ollama. It answers the two calls the app makes:
 *
 *   GET  /api/tags      -> the server is there
 *   POST /api/generate  -> one recorded reply (a JSON query plan), chosen by the question in the
 *                          prompt; any other question gets a plan that counts everything
 *
 * Run: node tests/e2e/helpers/fakeOllama.js [port]
 */
const http = require('http');

const PORT = Number(process.argv[2] || 11999);

/** Recorded replies of the local model, by a phrase of the question they answer. */
const RECORDED = [
  {
    when: 'came back positive',
    reply: {
      thought: 'Count patients with result positive and age_years at least 65, per visit_year.',
      answer: 'count',
      where: "age_years >= 65 AND result = 'Positive'",
      group_by: 'visit_year',
      numerator: null,
      number: null,
      outcome: null,
    },
  },
];
const DEFAULT_REPLY = {
  thought: 'Count everyone.',
  answer: 'count',
  where: null,
  group_by: null,
  numerator: null,
  number: null,
  outcome: null,
};

const server = http.createServer((req, res) => {
  const send = (status, body) => {
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body));
  };
  if (req.method === 'GET' && req.url === '/api/tags') {
    send(200, { models: [{ name: 'e2e-recorded' }] });
    return;
  }
  if (req.method === 'POST' && req.url === '/api/generate') {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
    });
    req.on('end', () => {
      let prompt;
      try {
        prompt = String(JSON.parse(raw).prompt || '');
      } catch {
        send(400, { error: 'not JSON' });
        return;
      }
      // The question is the last "Question:" of the prompt (examples come before it).
      const question = prompt.slice(prompt.lastIndexOf('Question:'));
      const found = RECORDED.find((entry) => question.includes(entry.when));
      send(200, { response: JSON.stringify(found ? found.reply : DEFAULT_REPLY), done: true });
    });
    return;
  }
  send(404, { error: 'not found' });
});

server.listen(PORT, '127.0.0.1');
