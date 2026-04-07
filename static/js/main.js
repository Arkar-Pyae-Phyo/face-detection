/**
 * =============================================================
 *  Exam Face Verification System – Shared JavaScript Utilities
 * =============================================================
 */

/**
 * Show a status message in the status bar.
 * @param {string} message - Text to display
 * @param {string} type    - 'info' | 'success' | 'error' | 'warning'
 */
function showStatus(message, type = 'info') {
  const bar = document.getElementById('statusBar');
  if (!bar) return;

  bar.className = `status-bar show ${type}`;
  bar.textContent = message;
}

/**
 * Clear / hide the status bar.
 */
function clearStatus() {
  const bar = document.getElementById('statusBar');
  if (bar) {
    bar.className = 'status-bar';
    bar.innerHTML = '';
  }
}

/**
 * Simple sleep utility for async/await.
 * @param {number} ms - milliseconds
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
