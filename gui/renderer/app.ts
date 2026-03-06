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

type ServerMessage = LogMessage | StatusMessage | { type: 'pong' } | { type: 'hide' } | { type: 'show' } | { type: 'transcript'; text: string };
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

const STATUS_LABELS: Record<string, string> = {
  idle: 'Idle', running: 'Running', done: 'Done', error: 'Error',
};

const CONN_LABELS: Record<ConnState, string> = {
  connected: 'Online', connecting: '...', disconnected: 'Offline',
};

// ── DOM ──────────────────────────────────────────────────────────────────

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const panel = $<HTMLDivElement>('panel');
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
const feed = $<HTMLDivElement>('feed');
const stepCountEl = $<HTMLSpanElement>('stepCount');

// ── State ────────────────────────────────────────────────────────────────

let ws: WebSocket | null = null;
let reconnectDelay = RECONNECT_BASE;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let pingTimer: ReturnType<typeof setInterval> | null = null;
let stepCount = 0;
let currentStatus = 'idle';

// ── Speech Recognition (Gemini Audio API via Python Backend) ─────────────

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
    // Attempt to use webm or just a generic audio format depending on platform support.
    const opts = MediaRecorder.isTypeSupported('audio/webm') ? { mimeType: 'audio/webm' } : undefined;
    mediaRecorder = new MediaRecorder(stream, opts);
    audioChunks = [];

    mediaRecorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
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

      // Stop all tracks to release the microphone lock
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

  const running = s === 'running';
  btnRun.disabled = running;
  btnStop.disabled = !running;
  progressTrack.classList.toggle('visible', running);
  panel.classList.toggle('running', running);

  if (s === 'done' || s === 'error') { btnRun.disabled = false; btnStop.disabled = true; }
}

function showStep(log: LogMessage): void {
  const levelClass = log.level === 'error' ? ' error' : log.level === 'success' ? ' success' : '';

  feed.innerHTML = `
    <div class="step-entry">
      <span class="step-icon">${esc(log.icon || '\u00b7')}</span>
      <span class="step-msg${levelClass}">${esc(log.message)}</span>
      <span class="step-time">${esc(log.time)}</span>
    </div>
  `;

  if (log.level === 'step' || log.level === 'success') {
    stepCount++;
    stepCountEl.textContent = String(stepCount);
  }
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
  feed.innerHTML = '';
  stepCount = 0;
  stepCountEl.textContent = '0';
  send({ type: 'start_task', task });
}

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

feed.innerHTML = '<span class="step-idle">Ready</span>';
connect();
