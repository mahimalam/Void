const { app, BrowserWindow, screen, Tray, Menu, nativeImage, ipcMain, shell, session } = require('electron');
const path = require('path');

let mainWindow;
let tray = null;

// Base64 generic blue circle icon for JARVIS tray (16x16)
const iconBase64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAGVJREFUOE9jZKAQMELp/1Aso2EwZGAgTBBhYGDACzAmhTADyDaMg10ByQZQDSCbQKoBVAPIJoRqwGAAWQ3D4AVQvUDQAGJcQKwB1HAA1QCSHUCyAUQ7gGQDSDaAaAcQ7QCSDSDaAQA83w4x/K3bJwAAAABJRU5ErkJggg==";

// Strict Content Security Policy for the HUD renderer.
// Local file:// for app assets + Google Fonts CDN for the Orbitron/JetBrains Mono fonts.
// No inline scripts (we removed the need for them when fixing C3).
// No eval, no remote module, no <object>/<embed>.
const HUD_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com data:",
  "img-src 'self' data:",
  "connect-src 'self' ws://127.0.0.1:9001",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'",
].join('; ');

function isSafeExternalUrl(rawUrl) {
  if (typeof rawUrl !== 'string') return false;
  try {
    const u = new URL(rawUrl);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return false;
    // Block SSRF-style loopback to anything other than the local event bus host
    if (u.hostname === '127.0.0.1' || u.hostname === 'localhost') {
      if (u.port && u.port !== '9001') return false;
    }
    return true;
  } catch (_) {
    return false;
  }
}

function createWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  // Upgraded dimensions for holographic Arc Reactor HUD
  const hudWidth = 540;
  const hudHeight = 270;

  mainWindow = new BrowserWindow({
    width: hudWidth,
    height: hudHeight,
    x: Math.max(20, width - hudWidth - 24), // Bottom right with margin
    y: Math.max(20, height - hudHeight - 24),
    frame: false,             // Frameless
    transparent: true,        // Transparent background
    hasShadow: true,
    alwaysOnTop: true,        // Overlay
    skipTaskbar: false,       // Accessible in taskbar on KDE/Wayland
    focusable: true,          // Allow interaction with buttons/links
    show: false,              // explicit show to override inherited SW_HIDE
    backgroundColor: '#00000000',
    webPreferences: {
      nodeIntegration: false,         // C3: no Node in renderer
      contextIsolation: true,         // C3: isolated preload world
      sandbox: false,                 // Required for --no-sandbox Linux compatibility
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: true,
      allowRunningInsecureContent: false,
    }
  });

  // Apply strict CSP via response headers — defence in depth alongside the
  // <meta http-equiv="Content-Security-Policy"> in index.html.
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [HUD_CSP],
        'X-Content-Type-Options': ['nosniff'],
      },
    });
  });

  mainWindow.loadFile('index.html');
  
  mainWindow.webContents.on('console-message', (event, ...args) => {
    const msg = (event && typeof event.message === 'string') ? event.message : (args[1] || event || '');
    console.log(`[HUD-Renderer] ${msg}`);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Fallback show after 300ms in case Wayland compositor delays ready-to-show
  setTimeout(() => {
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  }, 300);

  // Intercept any link clicks — open them in the system browser, not inside Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isSafeExternalUrl(url)) {
      shell.openExternal(url);
    }
    return { action: 'deny' }; // Prevent Electron from opening a new window
  });

  mainWindow.webContents.on('will-navigate', (event, url) => {
    // If it's NOT the local index.html, open it externally
    if (!url.startsWith('file://')) {
      event.preventDefault();
      if (isSafeExternalUrl(url)) {
        shell.openExternal(url);
      }
    }
  });
}

ipcMain.on('hud:minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.on('hud:hide', () => {
  if (mainWindow) mainWindow.hide();
});

// IPC: renderer asks main to open a URL. Validated here, not in the renderer.
ipcMain.handle('hud:open-external', async (_event, url) => {
  if (!isSafeExternalUrl(url)) {
    return { ok: false, error: 'refused' };
  }
  try {
    await shell.openExternal(url);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: String(e) };
  }
});

// H36: the renderer asks main to confirm the backend has shut
// down. We forward the request to the renderer so it can send
// ``{"command": "shutdown"}`` over the WebSocket, then wait
// for the WebSocket connection to close (which the Python
// ``start_websocket_server`` ``stop()`` method triggers when
// the orchestrator's cleanup block runs).
ipcMain.handle('hud:shutdown-backend', async (_event, timeoutMs = 8000) => {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (status) => {
      if (settled) return;
      settled = true;
      resolve({ ok: status === 'ok', status });
    };

    // Trigger the renderer to send the shutdown command. We do
    // this via the renderer because it owns the WebSocket
    // connection; the main process only has the BrowserWindow.
    if (mainWindow) {
      mainWindow.webContents.send('shutdown-backend');
    }

    // Watch for the WebSocket close: when the Python backend
    // shuts its server down, the renderer sees a clean
    // ``onclose`` event. We can't directly observe that here,
    // so we use a heuristic — poll the renderer's
    // WebSocket state via ``executeJavaScript`` and resolve
    // once it's ``CLOSED``.
    const deadline = Date.now() + Math.max(1000, timeoutMs);
    const poll = async () => {
      if (settled) return;
      if (Date.now() > deadline) {
        finish('timeout');
        return;
      }
      try {
        if (mainWindow) {
          const readyState = await mainWindow.webContents.executeJavaScript(
            'window.__jarvisSocket ? window.__jarvisSocket.readyState : -1'
          );
          // WebSocket.CLOSED === 3
          if (readyState === 3) {
            finish('ok');
            return;
          }
        }
      } catch (e) {
        // Renderer may have been torn down already.
        finish('error');
        return;
      }
      setTimeout(poll, 100);
    };
    setTimeout(poll, 100);
  });
});

function createTray() {
  const icon = nativeImage.createFromDataURL(iconBase64);
  tray = new Tray(icon);
  tray.setToolTip('J.A.R.V.I.S.');

  const contextMenu = Menu.buildFromTemplate([
    { label: 'J.A.R.V.I.S. Background Service', enabled: false },
    { type: 'separator' },
    { label: 'Show HUD', click: () => {
        if (mainWindow) mainWindow.show();
    }},
    { label: 'Hide HUD', click: () => {
        if (mainWindow) mainWindow.hide();
    }},
    { type: 'separator' },
    { label: 'Quit', click: async () => {
        // H36: the previous implementation sent a "shutdown"
        // message to the backend and called ``app.quit()`` after
        // a fixed 1.5s timer. On a slow machine the timer could
        // expire before the backend finished its cleanup (the
        // daily compaction, DB close, lock-file removal), which
        // would lose data. We now:
        //   1. Ask the renderer to forward a shutdown command
        //      over the WebSocket.
        //   2. Wait for the main process to confirm the backend
        //      has shut down (with an 8s safety timeout).
        //   3. Only then call ``app.quit()``.
        try {
            if (mainWindow) {
                await mainWindow.webContents.executeJavaScript(
                    'window.electronAPI.shutdownBackendAndWait(8000)'
                );
            }
        } catch (e) {
            console.error('Shutdown handshake failed:', e);
        }
        app.exit(0);
    }}
  ]);

  tray.setContextMenu(contextMenu);
}

// 2026-06-20: single-instance lock. Without this, every
// `npm.cmd start` spawn (one per JARVIS launch, one per
// restart_jarvis.ps1, one per `python -m core.main`) opened
// a new BrowserWindow, leaving 4+ HUDs stacked on the screen.
// ``requestSingleInstanceLock`` is an OS-level lock: the second
// process detects the existing instance, signals it via the
// ``second-instance`` event (so we can focus the existing
// window), and exits. Net effect: exactly one HUD at all times.
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  // Another instance is already running. Hand off and exit.
  app.quit();
  return;
}

app.on('second-instance', () => {
  // The user (or a JARVIS restart) tried to launch a second HUD.
  // Bring the existing window forward instead of opening a new one.
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    if (!mainWindow.isVisible()) mainWindow.show();
    mainWindow.focus();
  }
});

app.whenReady().then(() => {
  createWindow();
  createTray();

  // 2026-06-20: write the HUD's PID to data/hud.pid so the
  // Python side can detect that an HUD is already running
  // and skip the npm spawn on JARVIS restart. The Python
  // ctypes-based detection also checks for a visible
  // Chromium window as a cross-check. We write PID on
  // startup and clear it on quit.
  const fs = require('fs');
  const path = require('path');
  const pidFile = path.join(__dirname, '..', '..', 'data', 'hud.pid');
  try {
    fs.mkdirSync(path.dirname(pidFile), { recursive: true });
    fs.writeFileSync(pidFile, String(process.pid));
  } catch (e) {
    console.error('Failed to write hud.pid:', e);
  }
  app.on('will-quit', () => {
    try {
      const current = fs.readFileSync(pidFile, 'utf8').trim();
      if (current === String(process.pid)) {
        fs.unlinkSync(pidFile);
      }
    } catch (_) {
      // Best effort.
    }
  });

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
