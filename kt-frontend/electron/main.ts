import { app, BrowserWindow, Tray, Menu, nativeImage, globalShortcut, ipcMain, session } from 'electron'
import { join } from 'path'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import { homedir } from 'os'
import { registerIpcHandlers } from './ipc/handlers'
import { IPC_CHANNELS } from './ipc/channels'

const isDev = process.env.NODE_ENV === 'development'
const STATE_FILE = join(homedir(), '.ktransformers', 'window-state.json')

// ── Window state persistence ──────────────────────────────────────────────────
function loadWindowState(): { width: number; height: number; x?: number; y?: number } {
  try {
    if (existsSync(STATE_FILE)) {
      return JSON.parse(readFileSync(STATE_FILE, 'utf8'))
    }
  } catch {}
  return { width: 1400, height: 900 }
}

function saveWindowState(win: BrowserWindow) {
  try {
    if (win.isMaximized() || win.isMinimized()) return
    const bounds = win.getBounds()
    mkdirSync(join(homedir(), '.ktransformers'), { recursive: true })
    writeFileSync(STATE_FILE, JSON.stringify(bounds))
  } catch {}
}

// ── Tray ──────────────────────────────────────────────────────────────────────
let tray: Tray | null = null
let serverRunning = false

function buildTrayMenu(win: BrowserWindow): Menu {
  return Menu.buildFromTemplate([
    {
      label: `Server: ${serverRunning ? '● Running' : '○ Stopped'}`,
      enabled: false
    },
    { type: 'separator' },
    { label: 'Open Dashboard', click: () => { win.show(); win.focus() } },
    {
      label: serverRunning ? 'Stop Server' : 'Start Server',
      click: () => { win.show(); win.webContents.send(IPC_CHANNELS.NAV_GOTO, '/service') }
    },
    { type: 'separator' },
    { label: 'Quit', click: () => { app.quit() } }
  ])
}

function createTray(win: BrowserWindow) {
  // 16x16 coloured dot as tray icon (inline PNG base64)
  const stoppedIcon = nativeImage.createFromDataURL(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAGElEQVQ4T2NkIAIwjiYwGgajYRQMBgADCAABnlmSHQAAAABJRU5ErkJggg=='
  )
  const runningIcon = nativeImage.createFromDataURL(
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAHElEQVQ4T2Nk+M9Qz0AEYBxNYDQMRsMoGAwGAAWyAAHFbgOMAAAAAElFTkSuQmCC'
  )

  tray = new Tray(stoppedIcon)
  tray.setToolTip('KTransformers')
  tray.setContextMenu(buildTrayMenu(win))

  tray.on('double-click', () => { win.show(); win.focus() })

  // Update tray when server state changes from renderer
  ipcMain.on('tray:server-state', (_, running: boolean) => {
    serverRunning = running
    tray?.setImage(running ? runningIcon : stoppedIcon)
    tray?.setToolTip(`KTransformers — Server ${running ? 'Running' : 'Stopped'}`)
    tray?.setContextMenu(buildTrayMenu(win))
  })
}

// ── Main window ───────────────────────────────────────────────────────────────
function createWindow() {
  const state = loadWindowState()

  const win = new BrowserWindow({
    width: state.width,
    height: state.height,
    x: state.x,
    y: state.y,
    webPreferences: {
      preload: join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false  // allow cross-origin requests to remote API servers
    }
  })

  // Bypass proxy for loopback so local server connections work under VPN/proxy
  session.defaultSession.setProxy({
    proxyBypassRules: '<-loopback>'
  })

  registerIpcHandlers(win)
  createTray(win)

  // Save state on move/resize (debounced)
  let saveTimer: ReturnType<typeof setTimeout>
  const debouncedSave = () => {
    clearTimeout(saveTimer)
    saveTimer = setTimeout(() => saveWindowState(win), 500)
  }
  win.on('move', debouncedSave)
  win.on('resize', debouncedSave)
  win.on('close', () => saveWindowState(win))

  // ── Keyboard shortcuts (Ext 10) ────────────────────────────────────────────
  const routes = ['/', '/service', '/models', '/config', '/bench', '/chat']
  const mod = process.platform === 'darwin' ? 'Command' : 'Control'
  routes.forEach((route, i) => {
    globalShortcut.register(`${mod}+${i + 1}`, () => {
      win.show()
      win.webContents.send(IPC_CHANNELS.NAV_GOTO, route)
    })
  })

  if (isDev) {
    win.loadURL('http://localhost:5173')
    win.webContents.openDevTools()
  } else {
    win.loadFile(join(__dirname, '../dist/index.html'))
  }
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  globalShortcut.unregisterAll()
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})

app.on('will-quit', () => {
  globalShortcut.unregisterAll()
})
