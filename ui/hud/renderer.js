// J.A.R.V.I.S. HUD // MARK 85 Renderer Engine
// High-performance canvas Arc Reactor + Audio Spectrum + Live Telemetry

const hudRoot = document.getElementById('hud-root');
const coreWing = document.getElementById('core-wing');
const statusBadge = document.getElementById('status-badge');
const statusSub = document.getElementById('status-sub');
const toolStream = document.getElementById('tool-stream');

const cpuVal = document.getElementById('cpu-val');
const cpuBar = document.getElementById('cpu-bar');
const ramVal = document.getElementById('ram-val');
const ramBar = document.getElementById('ram-bar');
const authBadge = document.getElementById('auth-badge');
const authVal = document.getElementById('auth-val');

const latStt = document.getElementById('lat-stt');
const latAi = document.getElementById('lat-ai');
const latTts = document.getElementById('lat-tts');
const latTotal = document.getElementById('lat-total');

const btnPin = document.getElementById('btn-pin');
const btnGhost = document.getElementById('btn-ghost');
const btnMinimize = document.getElementById('btn-minimize');
const btnClose = document.getElementById('btn-close');

// Electron API bridge from preload
const api = window.electronAPI;

// Ghost Mode Toggle
let isGhostMode = false;
btnGhost.addEventListener('click', () => {
  isGhostMode = !isGhostMode;
  if (isGhostMode) {
    hudRoot.classList.add('ghost-mode');
    btnGhost.style.background = 'rgba(0, 240, 255, 0.3)';
  } else {
    hudRoot.classList.remove('ghost-mode');
    btnGhost.style.background = '';
  }
});

// Window controls
btnMinimize.addEventListener('click', () => {
  if (api && api.minimize) api.minimize();
});
btnClose.addEventListener('click', () => {
  if (api && api.hide) api.hide();
});

// -------------------------------------------------------------
// Pin Mode & Non-Intrusive Materialization State Engine
// -------------------------------------------------------------
let isPinned = localStorage.getItem('void_hud_pinned') === 'true';
let dematerializeTimer = null;

function updatePinUI() {
  if (!btnPin) return;
  if (isPinned) {
    btnPin.innerText = 'PIN: ON';
    btnPin.classList.add('active');
    btnPin.setAttribute('title', 'HUD is Pinned (Always Visible). Click to enable Auto-Hide.');
  } else {
    btnPin.innerText = 'PIN: AUTO';
    btnPin.classList.remove('active');
    btnPin.setAttribute('title', 'HUD Auto-Hides on Standby. Click to Pin permanently.');
  }
}

function materializeHUD(pulse = true) {
  if (dematerializeTimer) {
    clearTimeout(dematerializeTimer);
    dematerializeTimer = null;
  }
  hudRoot.classList.remove('dormant');
  hudRoot.classList.add('materialized');
  if (pulse) {
    hudRoot.classList.add('wake-active');
    setTimeout(() => hudRoot.classList.remove('wake-active'), 1400);
  }
  if (api && api.setClickThrough) {
    api.setClickThrough(false);
  }
  if (api && api.showInactive) {
    api.showInactive();
  }
}

function dematerializeHUD() {
  if (isPinned) return;
  hudRoot.classList.remove('materialized', 'wake-active');
  hudRoot.classList.add('dormant');
  if (api && api.setClickThrough) {
    api.setClickThrough(true);
  }
}

function scheduleDematerialize(delayMs = 2400) {
  if (isPinned) return;
  if (dematerializeTimer) clearTimeout(dematerializeTimer);
  dematerializeTimer = setTimeout(() => {
    dematerializeTimer = null;
    if (currentState === 'idle') {
      dematerializeHUD();
    }
  }, delayMs);
}

if (btnPin) {
  updatePinUI();
  btnPin.addEventListener('click', () => {
    isPinned = !isPinned;
    localStorage.setItem('void_hud_pinned', isPinned ? 'true' : 'false');
    updatePinUI();
    if (isPinned) {
      materializeHUD(false);
    } else if (currentState === 'idle') {
      dematerializeHUD();
    }
  });
}

// Initial state on startup
if (isPinned) {
  materializeHUD(false);
} else {
  dematerializeHUD();
}

// Listen for global summon shortcut from main process
if (api && api.onWake) {
  api.onWake(() => {
    materializeHUD(true);
    triggerWakeSequence();
    addToolLog('Summon triggered', 'running');
  });
}

// Safe external links
document.addEventListener('click', (e) => {
  const link = e.target.closest('a[href]');
  if (link && link.href && !link.href.startsWith('file://')) {
    e.preventDefault();
    if (api && api.openExternal) api.openExternal(link.href);
  }
});

// -------------------------------------------------------------
// State Machine & Telemetry
// -------------------------------------------------------------
let currentState = 'idle'; // idle | wake | listening | thinking | speaking | error | disconnected
let stateStartTime = Date.now();

const STATE_CONFIG = {
  idle: { badge: 'SLEEPING', sub: 'SAY "VOID" • READY', color: '#00F0FF', spinSpeed: 0.008, pulseSpeed: 0.003 },
  wake: { badge: 'WAKE', sub: 'CORE ONLINE // SENSORS ENGAGED', color: '#FFFFFF', spinSpeed: 0.06, pulseSpeed: 0.02 },
  listening: { badge: 'LISTENING', sub: 'ACOUSTIC RECEPTORS ACTIVE', color: '#00FF9D', spinSpeed: 0.022, pulseSpeed: 0.015 },
  transcribing: { badge: 'TRANSCRIBING', sub: 'NEURAL STREAM PARSING', color: '#00F0FF', spinSpeed: 0.04, pulseSpeed: 0.015 },
  thinking: { badge: 'PROCESSING', sub: 'NEURAL INFERENCE ACTIVE', color: '#00F0FF', spinSpeed: 0.05, pulseSpeed: 0.02 },
  speaking: { badge: 'SPEAKING', sub: 'AUDIO SYNTHESIS STREAM', color: '#BD00FF', spinSpeed: 0.03, pulseSpeed: 0.025 },
  error: { badge: 'SYS.FAULT', sub: 'SYSTEM ANOMALY DETECTED', color: '#FF3366', spinSpeed: 0.012, pulseSpeed: 0.01 },
  disconnected: { badge: 'OFFLINE', sub: 'RECONNECTING TO CORE...', color: '#8899A6', spinSpeed: 0.004, pulseSpeed: 0.002 }
};

function setState(stateName) {
  currentState = stateName;
  stateStartTime = Date.now();
  const cfg = STATE_CONFIG[stateName] || STATE_CONFIG.idle;

  statusBadge.innerText = cfg.badge;
  statusSub.innerText = cfg.sub;

  const isDormant = hudRoot.classList.contains('dormant');
  const isMaterialized = hudRoot.classList.contains('materialized');
  const isWakeActive = hudRoot.classList.contains('wake-active');

  // Clear previous state classes and assign current while preserving overlay state
  coreWing.className = `core-wing state-${stateName}`;
  hudRoot.className = `hud-chassis state-${stateName}${isGhostMode ? ' ghost-mode' : ''}${isMaterialized ? ' materialized' : ''}${isDormant ? ' dormant' : ''}${isWakeActive ? ' wake-active' : ''}`;
}

function addToolLog(text, tagType = 'running') {
  const el = document.createElement('div');
  el.className = `tool-entry ${tagType}`;

  const tagSpan = document.createElement('span');
  tagSpan.className = 'tag';
  tagSpan.innerText = tagType === 'success' ? '[OK]' : tagType === 'error' ? '[ERR]' : '[EXEC]';

  const textSpan = document.createElement('span');
  textSpan.innerText = text;

  el.appendChild(tagSpan);
  el.appendChild(textSpan);
  toolStream.appendChild(el);

  // Keep only the last 4 log entries for clean HUD presentation
  while (toolStream.children.length > 4) {
    toolStream.removeChild(toolStream.firstChild);
  }
}

// -------------------------------------------------------------
// ARC REACTOR CANVAS ANIMATION ENGINE
// -------------------------------------------------------------
const arcCanvas = document.getElementById('arc-canvas');
const arcCtx = arcCanvas.getContext('2d');

let outerAngle = 0;
let innerAngle = 0;
let pulseTimer = 0;
let wakeFlash = 0;
let wakeLockTimer = null;
let pendingStateAfterWake = null;

function triggerWakeSequence() {
  setState('wake');
  wakeFlash = 1.0;
  stateStartTime = Date.now();
  if (wakeLockTimer) clearTimeout(wakeLockTimer);
  pendingStateAfterWake = null;
  wakeLockTimer = setTimeout(() => {
    wakeLockTimer = null;
    if (pendingStateAfterWake) {
      setState(pendingStateAfterWake);
      pendingStateAfterWake = null;
    }
  }, 1400);
}

if (arcCanvas) {
  arcCanvas.style.cursor = 'pointer';
  arcCanvas.setAttribute('title', 'Click Arc Reactor to wake J.A.R.V.I.S.');
  arcCanvas.addEventListener('click', () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ command: 'wake' }));
      addToolLog('Arc Reactor Tap-to-Talk triggered', 'running');
    }
    triggerWakeSequence();
  });
}

// Particles floating in reactor
const particles = Array.from({ length: 14 }, (_, i) => ({
  angle: (i / 14) * Math.PI * 2,
  radius: 26 + (i % 3) * 8,
  speed: 0.01 + (i % 4) * 0.005,
  size: 1 + (i % 2) * 0.8,
  alpha: 0.4 + (i % 3) * 0.2
}));

function drawArcReactor() {
  const width = arcCanvas.width;
  const height = arcCanvas.height;
  const cx = width / 2;
  const cy = height / 2;

  arcCtx.clearRect(0, 0, width, height);

  const cfg = STATE_CONFIG[currentState] || STATE_CONFIG.idle;
  const now = Date.now();
  
  // Speed & momentum
  outerAngle += cfg.spinSpeed;
  innerAngle -= cfg.spinSpeed * 1.35;
  pulseTimer += cfg.pulseSpeed;

  // Pulse modulation
  let pulse = Math.sin(pulseTimer * 10) * 0.5 + 0.5; // 0..1
  if (currentState === 'wake') {
    wakeFlash = Math.max(0, 1 - (now - stateStartTime) / 1200);
    pulse = Math.max(pulse, wakeFlash);
  } else {
    wakeFlash = 0;
  }

  // 1. Center Glow Radial Gradient
  const coreRadius = 24 + pulse * 4;
  const glowGrad = arcCtx.createRadialGradient(cx, cy, 2, cx, cy, 58);
  if (currentState === 'error') {
    glowGrad.addColorStop(0, 'rgba(255, 51, 102, 0.9)');
    glowGrad.addColorStop(0.4, 'rgba(255, 51, 102, 0.35)');
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  } else if (currentState === 'speaking') {
    glowGrad.addColorStop(0, 'rgba(230, 200, 255, 0.95)');
    glowGrad.addColorStop(0.35, 'rgba(189, 0, 255, 0.45)');
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  } else if (currentState === 'listening') {
    glowGrad.addColorStop(0, 'rgba(200, 255, 230, 0.95)');
    glowGrad.addColorStop(0.35, 'rgba(0, 255, 157, 0.45)');
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  } else {
    glowGrad.addColorStop(0, `rgba(255, 255, 255, ${0.8 + wakeFlash * 0.2})`);
    glowGrad.addColorStop(0.3, `rgba(0, 240, 255, ${0.45 + pulse * 0.2})`);
    glowGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
  }

  arcCtx.fillStyle = glowGrad;
  arcCtx.beginPath();
  arcCtx.arc(cx, cy, 58, 0, Math.PI * 2);
  arcCtx.fill();

  // 2. Outer Graduation Ring (Dial with ticks)
  arcCtx.save();
  arcCtx.translate(cx, cy);
  arcCtx.rotate(outerAngle);

  // Outer border
  arcCtx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
  arcCtx.lineWidth = 1;
  arcCtx.beginPath();
  arcCtx.arc(0, 0, 56, 0, Math.PI * 2);
  arcCtx.stroke();

  // Dial Ticks
  const numTicks = 36;
  for (let i = 0; i < numTicks; i++) {
    const angle = (i / numTicks) * Math.PI * 2;
    const isMajor = i % 9 === 0;
    const isMid = i % 3 === 0;
    const len = isMajor ? 5 : isMid ? 3 : 1.5;
    const r1 = 56;
    const r2 = 56 - len;

    arcCtx.strokeStyle = isMajor ? cfg.color : 'rgba(0, 240, 255, 0.4)';
    arcCtx.lineWidth = isMajor ? 1.5 : 1;
    arcCtx.beginPath();
    arcCtx.moveTo(Math.cos(angle) * r1, Math.sin(angle) * r1);
    arcCtx.lineTo(Math.cos(angle) * r2, Math.sin(angle) * r2);
    arcCtx.stroke();
  }
  arcCtx.restore();

  // 3. Stator Ring: 10 Electromagnetic Coils
  const numCoils = 10;
  const coilRadius = 45;
  for (let i = 0; i < numCoils; i++) {
    const angle = (i / numCoils) * Math.PI * 2 + innerAngle;
    const coilW = 8;
    const coilH = 6;

    arcCtx.save();
    arcCtx.translate(cx, cy);
    arcCtx.rotate(angle);
    arcCtx.translate(coilRadius, 0);

    // Stator copper / illuminated wedge
    arcCtx.fillStyle = currentState === 'error' ? 'rgba(255, 51, 102, 0.8)' : 'rgba(0, 240, 255, 0.75)';
    arcCtx.shadowColor = cfg.color;
    arcCtx.shadowBlur = 6;
    arcCtx.fillRect(-coilW / 2, -coilH / 2, coilW, coilH);

    // Wire winding overlay lines
    arcCtx.strokeStyle = '#060A14';
    arcCtx.lineWidth = 1;
    arcCtx.beginPath();
    arcCtx.moveTo(-1, -coilH / 2);
    arcCtx.lineTo(-1, coilH / 2);
    arcCtx.moveTo(2, -coilH / 2);
    arcCtx.lineTo(2, coilH / 2);
    arcCtx.stroke();

    arcCtx.restore();
  }

  // 4. Counter-Rotating Dashed Flux Ring
  arcCtx.save();
  arcCtx.translate(cx, cy);
  arcCtx.rotate(innerAngle * 1.5);
  arcCtx.strokeStyle = cfg.color;
  arcCtx.lineWidth = 1.5;
  arcCtx.setLineDash([6, 6]);
  arcCtx.beginPath();
  arcCtx.arc(0, 0, 36, 0, Math.PI * 2);
  arcCtx.stroke();
  arcCtx.restore();

  // 5. Floating Energy Particles
  particles.forEach(p => {
    p.angle += p.speed * (currentState === 'thinking' ? 2.5 : 1);
    const px = cx + Math.cos(p.angle) * p.radius;
    const py = cy + Math.sin(p.angle) * p.radius;

    arcCtx.fillStyle = cfg.color;
    arcCtx.shadowColor = cfg.color;
    arcCtx.shadowBlur = 4;
    arcCtx.globalAlpha = p.alpha * (0.6 + pulse * 0.4);
    arcCtx.beginPath();
    arcCtx.arc(px, py, p.size, 0, Math.PI * 2);
    arcCtx.fill();
    arcCtx.globalAlpha = 1.0;
  });

  // 6. Center Vibranium Core Circle with Triad / Inner Segments
  arcCtx.save();
  arcCtx.translate(cx, cy);

  // Inner boundary ring
  arcCtx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
  arcCtx.shadowColor = cfg.color;
  arcCtx.shadowBlur = 10;
  arcCtx.lineWidth = 2;
  arcCtx.beginPath();
  arcCtx.arc(0, 0, coreRadius * 0.75, 0, Math.PI * 2);
  arcCtx.stroke();

  // Center triangle / triad core
  arcCtx.rotate(outerAngle * 0.8);
  arcCtx.strokeStyle = cfg.color;
  arcCtx.lineWidth = 1.5;
  arcCtx.beginPath();
  for (let i = 0; i < 3; i++) {
    const triAngle = (i / 3) * Math.PI * 2;
    const tr = coreRadius * 0.55;
    const tx = Math.cos(triAngle) * tr;
    const ty = Math.sin(triAngle) * tr;
    if (i === 0) arcCtx.moveTo(tx, ty);
    else arcCtx.lineTo(tx, ty);
  }
  arcCtx.closePath();
  arcCtx.stroke();

  // Bright Center Nucleus
  arcCtx.fillStyle = '#FFFFFF';
  arcCtx.shadowColor = '#FFFFFF';
  arcCtx.shadowBlur = 12;
  arcCtx.beginPath();
  arcCtx.arc(0, 0, 4 + pulse * 2, 0, Math.PI * 2);
  arcCtx.fill();

  arcCtx.restore();

  requestAnimationFrame(drawArcReactor);
}
requestAnimationFrame(drawArcReactor);

// -------------------------------------------------------------
// BOTTOM AUDIO SPECTRUM VISUALIZER
// -------------------------------------------------------------
const specCanvas = document.getElementById('spectrum-canvas');
const specCtx = specCanvas.getContext('2d');
const NUM_BARS = 36;
let barHeights = new Float32Array(NUM_BARS).fill(2);

function drawSpectrum() {
  const w = specCanvas.width;
  const h = specCanvas.height;
  specCtx.clearRect(0, 0, w, h);

  const barWidth = (w / NUM_BARS) - 2;
  const now = Date.now();

  for (let i = 0; i < NUM_BARS; i++) {
    let target = 2;

    if (currentState === 'speaking') {
      // Harmonic voice frequencies
      const freq1 = Math.sin(now * 0.015 + i * 0.4);
      const freq2 = Math.cos(now * 0.009 + i * 0.8);
      target = Math.max(3, (freq1 * 0.5 + freq2 * 0.5 + 1) * 7.5);
    } else if (currentState === 'listening') {
      // Acoustic input fluctuation
      const noise = Math.sin(now * 0.02 + i * 0.6);
      target = Math.max(2, (noise + 1) * 4.5);
    } else if (currentState === 'thinking') {
      // Scanning wave across spectrum
      const wave = Math.sin((now * 0.008) - (i * 0.25));
      target = Math.max(2, (wave + 1) * 5);
    } else {
      // Ambient calm pulse
      target = 2 + Math.sin(now * 0.002 + i * 0.2) * 1.2;
    }

    // Smooth lerp
    barHeights[i] += (target - barHeights[i]) * 0.25;

    const x = i * (barWidth + 2);
    const bh = Math.min(h, Math.max(2, barHeights[i]));
    const y = h - bh;

    // Stark gradient
    const grad = specCtx.createLinearGradient(0, y, 0, h);
    if (currentState === 'speaking') {
      grad.addColorStop(0, '#BD00FF');
      grad.addColorStop(1, '#0077FF');
    } else if (currentState === 'listening') {
      grad.addColorStop(0, '#00FF9D');
      grad.addColorStop(1, '#00F0FF');
    } else if (currentState === 'error') {
      grad.addColorStop(0, '#FF3366');
      grad.addColorStop(1, '#660022');
    } else {
      grad.addColorStop(0, '#00F0FF');
      grad.addColorStop(1, '#004488');
    }

    specCtx.fillStyle = grad;
    specCtx.fillRect(x, y, barWidth, bh);
  }

  requestAnimationFrame(drawSpectrum);
}
requestAnimationFrame(drawSpectrum);

// -------------------------------------------------------------
// WEBSOCKET INTEGRATION WITH JARVIS CORE (Port 9001)
// -------------------------------------------------------------
let socket = null;
let reconnectTimer = null;
let reconnectDelay = 1000;

if (api && typeof api.onShutdownBackend === 'function') {
  api.onShutdownBackend(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ command: 'shutdown' }));
    }
  });
}

function connect() {
  socket = new WebSocket('ws://127.0.0.1:9001');
  window.__jarvisSocket = socket;

  socket.onopen = () => {
    console.log('[HUD] Connected to J.A.R.V.I.S. Core');
    reconnectDelay = 1000;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    setState('idle');
    addToolLog('Core bus connected (port 9001)', 'success');
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      const type = data.type;
      const payload = data.payload || {};

      switch (type) {
        case 'idle':
          if (wakeLockTimer) {
            clearTimeout(wakeLockTimer);
            wakeLockTimer = null;
            pendingStateAfterWake = null;
          }
          setState('idle');
          scheduleDematerialize(2400);
          break;
        case 'wake':
          materializeHUD(true);
          triggerWakeSequence();
          addToolLog('Wake word detected', 'running');
          break;
        case 'listening':
          materializeHUD(false);
          if (wakeLockTimer) {
            pendingStateAfterWake = 'listening';
          } else {
            setState('listening');
          }
          break;
        case 'transcribing':
          materializeHUD(false);
          if (wakeLockTimer) {
            clearTimeout(wakeLockTimer);
            wakeLockTimer = null;
            pendingStateAfterWake = null;
          }
          setState('transcribing');
          break;
        case 'thinking':
          materializeHUD(false);
          if (wakeLockTimer) {
            clearTimeout(wakeLockTimer);
            wakeLockTimer = null;
            pendingStateAfterWake = null;
          }
          setState('thinking');
          break;
        case 'speaking':
          materializeHUD(false);
          if (wakeLockTimer) {
            clearTimeout(wakeLockTimer);
            wakeLockTimer = null;
            pendingStateAfterWake = null;
          }
          setState('speaking');
          break;
        case 'error':
          materializeHUD(false);
          if (wakeLockTimer) {
            clearTimeout(wakeLockTimer);
            wakeLockTimer = null;
            pendingStateAfterWake = null;
          }
          setState('error');
          if (payload.error_msg) addToolLog(payload.error_msg, 'error');
          break;

        case 'system_stats':
          if (payload.cpu !== undefined) {
            cpuVal.innerText = payload.cpu + '%';
            const cpuRatio = Math.min(1, Math.max(0.05, payload.cpu / 100));
            cpuBar.style.transform = `scaleX(${cpuRatio})`;
          }
          if (payload.ram !== undefined) {
            ramVal.innerText = payload.ram + 'GB';
            const ramRatio = Math.min(1, Math.max(0.1, parseFloat(payload.ram) / 32));
            ramBar.style.transform = `scaleX(${ramRatio})`;
          }
          break;

        case 'auth_change':
          const newState = payload.new_state || 'UNLOCKED';
          authVal.innerText = newState === 'UNLOCKED' ? 'OMEGA // UNLOCKED' : 'SEC.LOCKED // RESTRICTED';
          if (newState === 'UNLOCKED') {
            authBadge.className = 'security-badge';
          } else {
            authBadge.className = 'security-badge locked';
          }
          break;

        case 'tool_start':
          addToolLog(payload.tool_name || 'tool_executing', 'running');
          break;

        case 'tool_done':
          addToolLog(payload.tool_name || 'tool_complete', payload.success ? 'success' : 'error');
          break;

        case 'latency_report':
          if (payload.timings) {
            const stt = payload.timings['stt'] ? Math.round(payload.timings['stt'] * 1000) : 0;
            const ai = payload.timings['first_token'] ? Math.round(payload.timings['first_token'] * 1000) : 0;
            const total = payload.timings['total_to_first_audio'] ? Math.round(payload.timings['total_to_first_audio'] * 1000) : 0;
            const tts = (total > 0 && ai > 0 && stt > 0) ? Math.max(0, total - stt - ai) : 0;

            latStt.innerText = stt > 0 ? stt + ' ms' : '--';
            latAi.innerText = ai > 0 ? ai + ' ms' : '--';
            latTts.innerText = tts > 0 ? tts + ' ms' : '--';
            latTotal.innerText = total > 0 ? total + ' ms' : '--';

            // SLA Color Coding
            if (total > 2500) {
              latTotal.style.color = '#FF3366'; // Red (>2.5s)
            } else if (total > 1500) {
              latTotal.style.color = '#FFB700'; // Amber (1.5 - 2.5s)
            } else {
              latTotal.style.color = '#00FF9D'; // Green (<1.5s)
            }
          }
          break;
      }
    } catch (err) {
      console.error('[HUD] Error parsing event message:', err);
    }
  };

  socket.onclose = () => {
    console.log('[HUD] Disconnected from Core. Retrying in', reconnectDelay, 'ms');
    setState('disconnected');
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    connect();
    reconnectDelay = Math.min(reconnectDelay * 2, 20000);
  }, reconnectDelay);
}

// Initial start
connect();
