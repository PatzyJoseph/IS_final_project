// DOM Elements
const startButton = document.getElementById('startButton');
const disclaimerModal = document.getElementById('disclaimerModal');
const agreeButton = document.getElementById('agreeButton');
const cancelButton = document.getElementById('cancelButton');

// Event Listeners
startButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'flex';
});

cancelButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'none';
});

agreeButton.addEventListener('click', () => {
    disclaimerModal.style.display = 'none';
    // Redirect to the analysis page
    window.location.href = 'analysis.html';
});

// Close modal if clicking outside
disclaimerModal.addEventListener('click', (e) => {
    if (e.target === disclaimerModal) {
        disclaimerModal.style.display = 'none';
    }
});