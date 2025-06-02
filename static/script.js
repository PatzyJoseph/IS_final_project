// script.js
// DOM Elements
const startButton = document.getElementById('startButton');
const disclaimerModal = document.getElementById('disclaimerModal');
const agreeButton = document.getElementById('agreeButton');
const cancelButton = document.getElementById('cancelButton');

// Create and configure a hidden file input
const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = 'image/*';
fileInput.style.display = 'none';
document.body.appendChild(fileInput);

// Event Listeners
startButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'flex';
});

cancelButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'none';
});

agreeButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'none';
    // Redirect to analysis page
    window.location.href = 'analysis';
});

disclaimerModal.addEventListener('click', (e) => {
    if (e.target === disclaimerModal) {
        disclaimerModal.style.display = 'none';
    }
});

// When an image is selected
fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const json = await res.json();

        if (!res.ok) throw new Error(json.error || 'Unknown error occurred');

        // Show results in a nice way
        const resultMessage = `
          <h3>Analysis Result:</h3>
          <ul>
            <li>🩸 Bleeding: ${json.bleeding ? '✅ Detected' : '❌ Not Detected'}</li>
            <li>😮‍💨 Swelling: ${json.redness ? '✅ Detected' : '❌ Not Detected'}</li>
            <li>🔴 Redness: ${json.swelling ? '✅ Detected' : '❌ Not Detected'}</li>
          </ul>
          <p><strong>Gingivitis Diagnosis:</strong> ${json.gingivitis ? '🟥 Positive' : '✅ Negative'}</p>
        `;

        // Show in modal or inject into page
        showResult(resultMessage);
    } catch (err) {
        console.error(err);
        alert('Error: ' + err.message);
    } finally {
        fileInput.value = ''; // Reset input
    }
});

// Injects the result into the DOM
function showResult(messageHTML) {
    const existing = document.getElementById('resultBox');
    if (existing) existing.remove();

    const box = document.createElement('div');
    box.id = 'resultBox';
    box.innerHTML = messageHTML;
    box.style.padding = '1rem';
    box.style.margin = '2rem auto';
    box.style.border = '2px solid #ccc';
    box.style.borderRadius = '10px';
    box.style.maxWidth = '500px';
    box.style.backgroundColor = '#f8f8f8';
    box.style.boxShadow = '0 0 10px rgba(0,0,0,0.1)';
    box.style.fontSize = '1.1rem';

    document.body.appendChild(box);
}
