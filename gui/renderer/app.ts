export { };

// ── Types ────────────────────────────────────────────────────────────────

interface LogMessage {
  type: 'log';
  level: 'info' | 'step' | 'success' | 'warning' | 'error';
  message: string;
  time: string;
  icon: string;
}

interface StatusMessage {
  type: 'status';
  status: 'idle' | 'running' | 'done' | 'error';
}

interface ParsedStep {
  icon: string;
  agent: 'brain' | 'gui' | 'mcp' | 'infra' | 'code' | 'system';
  title: string;
  detail: string;
  level: string;
  timestamp: string;
}

interface TodoItem {
  id: number;
  text: string;
  status: 'pending' | 'in_progress' | 'done' | 'failed';
  subtasks?: TodoItem[];
}

type ServerMessage = LogMessage | StatusMessage | { type: 'pong' } | { type: 'hide' } | { type: 'show' } | { type: 'transcript'; text: string } | { type: 'todo_update'; items: TodoItem[] };
type ClientMessage = { type: 'start_task'; task: string } | { type: 'stop_task' } | { type: 'ping' } | { type: 'stt_audio'; data: string };
type ConnState = 'connected' | 'connecting' | 'disconnected';

declare global {
  interface Window {
    electronAPI?: { closeWindow: () => void; minimizeWindow: () => void; hideWindow: () => void; showWindow: () => void };
  }
}

const api = (window as any).electronAPI as
  | { closeWindow: () => void; minimizeWindow: () => void; hideWindow: () => void; showWindow: () => void }
  | undefined;

// ── Config ───────────────────────────────────────────────────────────────

const WS_URL = 'ws://localhost:7577/ws';
const RECONNECT_BASE = 1000;
const RECONNECT_MAX = 10000;
const PING_INTERVAL = 15000;
const MAX_HISTORY = 50;

const STATUS_LABELS: Record<string, string> = {
  idle: 'Idle', running: 'Running', done: 'Done', error: 'Error',
};

const CONN_LABELS: Record<ConnState, string> = {
  connected: 'Online', connecting: '...', disconnected: 'Offline',
};

const AGENT_LABELS: Record<ParsedStep['agent'], string> = {
  brain: 'Orchestrator', gui: 'GUI', mcp: 'MCP', infra: 'Infra', code: 'Code', system: 'System',
};

// ── DOM ──────────────────────────────────────────────────────────────────

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const island = $<HTMLDivElement>('island');
const pillIcon = $<HTMLSpanElement>('pillIcon');
const pillDot = $<HTMLSpanElement>('pillDot');
const pillLabel = $<HTMLSpanElement>('pillLabel');
const pillSteps = $<HTMLSpanElement>('pillSteps');
const feed = $<HTMLDivElement>('feed');
const taskInput = $<HTMLInputElement>('taskInput');
const btnRun = $<HTMLButtonElement>('btnRun');
const btnStop = $<HTMLButtonElement>('btnStop');
const btnMic = $<HTMLButtonElement>('btnMic');
const btnClose = $<HTMLButtonElement>('btnClose');
const btnMinimize = $<HTMLButtonElement>('btnMinimize');
const statusPill = $<HTMLElement>('statusPill');
const statusText = $<HTMLSpanElement>('statusText');
const connDot = $<HTMLElement>('connIndicator');
const connLabel = $<HTMLSpanElement>('connLabel');
const progressTrack = $<HTMLDivElement>('progressTrack');
const stepCountEl = $<HTMLSpanElement>('stepCount');
const todoListEl = $<HTMLDivElement>('todoList');

// ── State ────────────────────────────────────────────────────────────────

let ws: WebSocket | null = null;
let reconnectDelay = RECONNECT_BASE;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let pingTimer: ReturnType<typeof setInterval> | null = null;
let stepCount = 0;
let currentStatus = 'idle';
const stepHistory: ParsedStep[] = [];

// ── Message Parsing ──────────────────────────────────────────────────────

function mapNodeToAgent(node: string): ParsedStep['agent'] {
  if (node.includes('gui')) return 'gui';
  if (node.includes('mcp')) return 'mcp';
  if (node.includes('infra')) return 'infra';
  if (node.includes('code')) return 'code';
  return 'brain';
}

function prettifyAction(action: string): string {
  const a = action.toLowerCase().trim();
  if (a.startsWith('click')) return 'Clicking on screen element';
  if (a.startsWith('type')) return 'Typing text input';
  if (a.startsWith('scroll')) return 'Scrolling the view';
  if (a.startsWith('key')) return 'Pressing keyboard shortcut';
  if (a.startsWith('drag')) return 'Dragging element';
  if (a.startsWith('hover')) return 'Hovering over element';
  if (a.startsWith('wait')) return 'Waiting for page to load';
  if (a.startsWith('screenshot')) return 'Capturing screenshot';
  return action.charAt(0).toUpperCase() + action.slice(1);
}

function parseLogMessage(log: LogMessage): ParsedStep {
  const msg = log.message;
  let agent: ParsedStep['agent'] = 'system';
  let title = msg;
  let detail = '';

  if (/^Step \d+: Thinking/i.test(msg)) {
    agent = 'brain';
    const n = msg.match(/Step (\d+)/)?.[1] || '?';
    title = 'Analyzing current state';
    detail = `Examining the screen and deciding next action (step ${n})`;
  }
  else if (/^Capturing screen/i.test(msg)) {
    agent = 'brain';
    title = 'Capturing screenshot';
    detail = 'Taking a snapshot of the current desktop';
  }
  else if (/^→\s*(\w+):\s*(.+)/.test(msg)) {
    const match = msg.match(/^→\s*(\w+):\s*(.+)/);
    const node = match![1];
    const instruction = match![2];
    agent = mapNodeToAgent(node);
    title = `Delegating to ${AGENT_LABELS[agent]}`;
    detail = instruction;
  }
  else if (/^GUI Task:\s*(.+)/i.test(msg)) {
    agent = 'gui';
    title = msg.match(/^GUI Task:\s*(.+)/i)![1];
    detail = 'GUI agent navigating the screen';
  }
  else if (/^GUI Step \d+: Thinking/i.test(msg)) {
    agent = 'gui';
    const n = msg.match(/GUI Step (\d+)/)?.[1] || '?';
    title = `Visual analysis (sub-step ${n})`;
    detail = 'Examining screen elements for interaction';
  }
  else if (/^Action:\s*(.+)/i.test(msg)) {
    agent = 'gui';
    const action = msg.match(/^Action:\s*(.+)/i)![1];
    title = prettifyAction(action);
    detail = action !== title ? action : '';
  }
  else if (/^GUI: Done/i.test(msg)) {
    agent = 'gui';
    title = 'Screen interaction completed';
    detail = 'GUI agent finished successfully';
  }
  else if (/^GUI: Failed/i.test(msg)) {
    agent = 'gui';
    title = 'Screen interaction failed';
    detail = 'GUI agent encountered an error';
  }
  else if (/^MCP:\s*(.+)/i.test(msg)) {
    agent = 'mcp';
    title = msg.match(/^MCP:\s*(.+)/i)![1];
    detail = 'Calling external service via MCP';
  }
  else if (/^MCP result:\s*(.+)/i.test(msg)) {
    agent = 'mcp';
    title = 'Service responded';
    detail = msg.match(/^MCP result:\s*(.+)/i)![1].substring(0, 150);
  }
  else if (/^Code Agent:\s*(.+)/i.test(msg)) {
    agent = 'code';
    title = msg.match(/^Code Agent:\s*(.+)/i)![1];
    detail = 'Executing code operation';
  }
  else if (/^Infra:\s*(.+)/i.test(msg)) {
    agent = 'infra';
    title = msg.match(/^Infra:\s*(.+)/i)![1];
    detail = 'Running infrastructure command';
  }
  else if (/^Task:\s*(.+)/i.test(msg)) {
    agent = 'brain';
    title = 'New task received';
    detail = msg.match(/^Task:\s*(.+)/i)![1];
  }
  else if (/^Finished/i.test(msg)) {
    agent = 'brain';
    title = 'All operations completed';
    detail = msg;
  }
  else if (/^Stopped/i.test(msg)) {
    agent = 'system';
    title = 'Task stopped by user';
    detail = '';
  }
  else {
    if (log.icon === '') agent = 'brain';
    else if (log.icon === '') agent = 'gui';
    else if (log.icon === '') agent = 'mcp';
    else if (log.icon === '') agent = 'code';
    else if (log.icon === '' || log.icon === '') agent = 'infra';
    title = msg;
  }

  return { icon: log.icon || '\u00b7', agent, title, detail, level: log.level, timestamp: log.time };
}

// ── Speech Recognition ───────────────────────────────────────────────────

let mediaRecorder: MediaRecorder | null = null;
let audioChunks: Blob[] = [];
let isRecording = false;

async function toggleMic(): Promise<void> {
  if (isRecording && mediaRecorder) {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const opts = MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : undefined;
    mediaRecorder = new MediaRecorder(stream, opts);
    audioChunks = [];

    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener('stop', () => {
      isRecording = false;
      btnMic.classList.remove('recording');
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = () => {
        const base64Audio = (reader.result as string).split(',')[1];
        send({ type: 'stt_audio', data: base64Audio });
      };
      stream.getTracks().forEach(track => track.stop());
    });

    mediaRecorder.start();
    isRecording = true;
    btnMic.classList.add('recording');
    taskInput.value = 'Listening...';
  } catch (error) {
    console.error('Error accessing microphone:', error);
    shake(taskInput);
    taskInput.value = 'Microphone error.';
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────

function connect(): void {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  setConn('connecting');
  ws = new WebSocket(WS_URL);
  ws.onopen = () => { reconnectDelay = RECONNECT_BASE; setConn('connected'); startPing(); };
  ws.onmessage = (e: MessageEvent) => { try { handle(JSON.parse(e.data)); } catch { } };
  ws.onclose = () => { setConn('disconnected'); stopPing(); scheduleReconnect(); };
  ws.onerror = () => { };
}

function scheduleReconnect(): void {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    reconnectDelay = Math.min(reconnectDelay * 1.5, RECONNECT_MAX);
    connect();
  }, reconnectDelay);
}

function send(msg: ClientMessage): void {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function startPing(): void { stopPing(); pingTimer = setInterval(() => send({ type: 'ping' }), PING_INTERVAL); }
function stopPing(): void { if (pingTimer) { clearInterval(pingTimer); pingTimer = null; } }

// ── Message handler ──────────────────────────────────────────────────────

function handle(msg: ServerMessage): void {
  if (msg.type === 'log') showStep(msg);
  else if (msg.type === 'status') setStatus(msg.status);
  else if (msg.type === 'hide') api?.hideWindow();
  else if (msg.type === 'show') api?.showWindow();
  else if (msg.type === 'transcript') {
    taskInput.value = msg.text;
    taskInput.focus();
    startTask();
  }
  else if (msg.type === 'todo_update') {
    renderTodoList(msg.items);
  }
}

function renderTodoList(items: TodoItem[]) {
  if (!items || items.length === 0) {
    todoListEl.style.display = 'none';
    return;
  }
  todoListEl.style.display = 'flex';
  todoListEl.innerHTML = '';
  items.forEach(item => {
    todoListEl.appendChild(createTodoElement(item));
  });
}

function createTodoElement(item: TodoItem, depth: number = 0): HTMLElement {
  const container = document.createElement('div');
  container.className = `todo-container depth-${depth}`;

  const itemRow = document.createElement('div');
  itemRow.className = `todo-item status-${item.status}`;
  if (item.subtasks && item.subtasks.length > 0) {
    itemRow.classList.add('has-children');
    itemRow.onclick = () => {
      container.classList.toggle('collapsed');
    };
  }

  const icon = document.createElement('span');
  icon.className = 'todo-icon';
  if (item.status === 'done') icon.innerHTML = '✅';
  else if (item.status === 'in_progress') icon.innerHTML = '🔄';
  else if (item.status === 'failed') icon.innerHTML = '❌';
  else icon.innerHTML = '⬜';

  const text = document.createElement('span');
  text.className = 'todo-text';
  text.textContent = item.text;

  itemRow.appendChild(icon);
  itemRow.appendChild(text);

  if (item.subtasks && item.subtasks.length > 0) {
    const arrow = document.createElement('span');
    arrow.className = 'todo-arrow';
    arrow.innerHTML = '▾';
    itemRow.appendChild(arrow);
  }

  container.appendChild(itemRow);

  if (item.subtasks && item.subtasks.length > 0) {
    const childrenContainer = document.createElement('div');
    childrenContainer.className = 'todo-children';
    item.subtasks.forEach(sub => {
      childrenContainer.appendChild(createTodoElement(sub, depth + 1));
    });
    container.appendChild(childrenContainer);
  }

  return container;
}

// ── UI ───────────────────────────────────────────────────────────────────

function setConn(s: ConnState): void {
  connDot.setAttribute('data-state', s);
  connLabel.textContent = CONN_LABELS[s];
}

function setStatus(s: string): void {
  currentStatus = s;
  statusPill.setAttribute('data-status', s);
  statusText.textContent = STATUS_LABELS[s] || s;
  pillDot.setAttribute('data-status', s);

  const running = s === 'running';
  btnRun.disabled = running;
  btnStop.disabled = !running;
  progressTrack.classList.toggle('visible', running);
  island.classList.toggle('running', running);

  if (s === 'done' || s === 'error') {
    btnRun.disabled = false;
    btnStop.disabled = true;
  }

  if (s === 'done') {
    pillIcon.textContent = '\u2705';
    pillLabel.textContent = 'Task completed';
  } else if (s === 'error') {
    pillIcon.textContent = '\u274c';
    pillLabel.textContent = 'Task failed';
  } else if (s === 'idle') {
    pillIcon.textContent = '\u00b7';
    pillLabel.textContent = 'Ready';
  }
}

function showStep(log: LogMessage): void {
  const parsed = parseLogMessage(log);
  stepHistory.push(parsed);
  if (stepHistory.length > MAX_HISTORY) stepHistory.shift();

  // Update collapsed pill
  pillIcon.textContent = parsed.icon || '\u00b7';
  pillLabel.textContent = parsed.title;

  if (log.level === 'step' || log.level === 'success') {
    stepCount++;
    stepCountEl.textContent = String(stepCount);
    pillSteps.textContent = `Step ${stepCount}`;
  }

  // Append card to feed
  const levelClass = parsed.level === 'error' ? ' error' : parsed.level === 'success' ? ' success' : '';
  const card = document.createElement('div');
  card.className = 'step-card';
  card.innerHTML = `
    <span class="card-icon">${esc(parsed.icon || '\u00b7')}</span>
    <div class="card-body">
      <span class="card-agent" data-agent="${esc(parsed.agent)}">${esc(AGENT_LABELS[parsed.agent])}</span>
      <span class="card-title${levelClass}">${esc(parsed.title)}</span>
      ${parsed.detail ? `<span class="card-detail">${esc(parsed.detail)}</span>` : ''}
    </div>
    <span class="card-time">${esc(parsed.timestamp)}</span>
  `;
  feed.appendChild(card);
  feed.scrollTop = feed.scrollHeight;
}

function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function shake(el: HTMLElement): void {
  el.classList.remove('shake');
  void el.offsetWidth;
  el.classList.add('shake');
}

// ── Actions ──────────────────────────────────────────────────────────────

function startTask(): void {
  const task = taskInput.value.trim();
  if (!task) { shake(taskInput); taskInput.focus(); return; }

  stepHistory.length = 0;
  feed.innerHTML = '';
  stepCount = 0;
  stepCountEl.textContent = '0';
  pillSteps.textContent = '';
  pillIcon.textContent = '\uD83D\uDE80';
  pillLabel.textContent = 'Starting...';

  send({ type: 'start_task', task });
}

// ── Pin island open while input is focused ───────────────────────────────

taskInput.addEventListener('focus', () => island.classList.add('pinned'));
taskInput.addEventListener('blur', () => {
  setTimeout(() => {
    if (document.activeElement !== taskInput) island.classList.remove('pinned');
  }, 150);
});

// ── Events ───────────────────────────────────────────────────────────────

btnRun.addEventListener('click', startTask);
btnStop.addEventListener('click', () => send({ type: 'stop_task' }));
btnMic.addEventListener('click', toggleMic);
btnClose.addEventListener('click', () => api?.closeWindow());
btnMinimize.addEventListener('click', () => api?.minimizeWindow());
taskInput.addEventListener('keydown', (e: KeyboardEvent) => {
  if (e.key === 'Enter') { e.preventDefault(); startTask(); }
});

// ── Init ─────────────────────────────────────────────────────────────────

connect();
