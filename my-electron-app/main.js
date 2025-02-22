/**
 * @file main.js
 * @description This file is the main process of the Electron application.
 * It creates a browser window and loads the index.html file.
 */

// Import necessary modules from Electron and Node.js 'path'
const { app, BrowserWindow } = require("electron");
const path = require("path");

/**
 * @function createWindow
 * @description Creates and configures the main browser window for the application.
 * This function is responsible for setting the initial size of the window,
 * enabling Node.js integration, and loading the 'index.html' file.
 * @returns {void}
 */
function createWindow() {
	// Create a new browser window instance.
	const win = new BrowserWindow({
		width: 800, // Set the initial width of the window to 800 pixels.
		height: 600, // Set the initial height of the window to 600 pixels.
		webPreferences: {
			nodeIntegration: true, // Enable Node.js integration in the renderer process for accessing Node.js APIs.
		},
	});

	// Load the index.html file into the browser window.
	// This file is expected to be in the same directory as main.js or in a 'public' directory.
	win.loadFile("index.html");
}

// When the Electron app is ready (after initialisation), create the browser window.
// 'app.whenReady()' returns a Promise that resolves when Electron is ready to create browser windows.
app.whenReady().then(createWindow);