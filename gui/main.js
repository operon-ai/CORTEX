const { app, BrowserWindow, ipcMain, screen } = require('electron');
const path = require('path');

let mainWindow = null;

function createWindow() {
  const display = screen.getPrimaryDisplay();
  const { width: sw, height: sh } = display.workAreaSize;
  const winW = 340;
  const winH = 180;

  mainWindow = new BrowserWindow({
    width: winW,
    height: winH,
    x: sw - winW - 16,
    y: sh - winH - 16,
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

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  mainWindow.on('closed', () => { mainWindow = null; });
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());

ipcMain.on('close-window', () => mainWindow && mainWindow.close());
ipcMain.on('minimize-window', () => mainWindow && mainWindow.minimize());
ipcMain.on('hide-window', () => mainWindow && mainWindow.setOpacity(0));
ipcMain.on('show-window', () => mainWindow && mainWindow.setOpacity(1));
