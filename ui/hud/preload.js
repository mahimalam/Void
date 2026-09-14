// Preload script — runs in an isolated world before the renderer.
// Exposes a minimal, validated bridge to the renderer via contextBridge.
// The renderer CANNOT `require('electron')`; it can only use window.electronAPI.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  /**
   * Subscribe to the 'shutdown-backend' IPC event from the main process.
   * Returns an unsubscribe function so the renderer can clean up if it
   * ever wants to.
   */
  onShutdownBackend(callback) {
    if (typeof callback !== 'function') return () => {};
    const listener = () => callback();
    ipcRenderer.on('shutdown-backend', listener);
    return () => ipcRenderer.removeListener('shutdown-backend', listener);
  },

  /**
   * Ask the main process to open a URL in the system browser.
   * Main process validates the protocol (http/https only) before opening.
   */
  openExternal(url) {
    return ipcRenderer.invoke('hud:open-external', url);
  },

  /**
   * H36: ask the main process to terminate the backend AND
   * block until the main process has either confirmed the
   * termination or timed out. The renderer awaits this
   * promise before calling ``app.quit()`` so the backend
   * has time to flush its compaction / DB / lock-file
   * cleanup before the Python process disappears.
   */
  shutdownBackendAndWait(timeoutMs = 8000) {
    return ipcRenderer.invoke('hud:shutdown-backend', timeoutMs);
  },

  minimize() {
    ipcRenderer.send('hud:minimize');
  },

  hide() {
    ipcRenderer.send('hud:hide');
  },

  setClickThrough(enable) {
    ipcRenderer.send('hud:set-click-through', Boolean(enable));
  },

  showInactive() {
    ipcRenderer.send('hud:show-inactive');
  },

  onWake(callback) {
    if (typeof callback !== 'function') return () => {};
    const listener = () => callback();
    ipcRenderer.on('hud:wake-event', listener);
    return () => ipcRenderer.removeListener('hud:wake-event', listener);
  }
});
