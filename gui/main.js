const { app, BrowserWindow, ipcMain, screen, session } = require('electron');
const path = require('path');

let mainWindow = null;

function createWindow() {
  const display = screen.getPrimaryDisplay();
  const { width: sw, height: sh } = display.workAreaSize;
  const winW = 860;
  const winH = 460;

  mainWindow = new BrowserWindow({
    width: winW,
    height: winH,
    x: Math.round((sw - winW) / 2),
    y: 12,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: false,
    hasShadow: false,
    backgroundColor: '#00000000',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Prevent click-through: the window captures all mouse events within its bounds.
  // Forward-ignore is OFF — the island handles its own hit-testing via CSS.
  mainWindow.setIgnoreMouseEvents(false);

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  mainWindow.on('closed', () => { mainWindow = null; });
}

app.whenReady().then(() => {
  // Auto-grant microphone permission for Web Speech API
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === 'media') {
      callback(true);
    } else {
      callback(false);
    }
  });
  createWindow();
});
app.on('window-all-closed', () => app.quit());

ipcMain.on('close-window', () => mainWindow && mainWindow.close());
ipcMain.on('minimize-window', () => mainWindow && mainWindow.minimize());
ipcMain.on('hide-window', () => mainWindow && mainWindow.setOpacity(0));
ipcMain.on('show-window', () => mainWindow && mainWindow.setOpacity(1));
