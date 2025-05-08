// DOM Elements
const fileUpload = document.getElementById('fileUpload');
const fileUploadContainer = document.getElementById('fileUploadContainer');
const imagePreview = document.getElementById('imagePreview');
const uploadedImage = document.getElementById('uploadedImage');
const submitButton = document.getElementById('submitButton');
const analysisResult = document.getElementById('analysisResult');
const resultImage = document.querySelector('.result-image');
const tryAgainButton = document.querySelector('.try-again-button');

// Handle file selection
fileUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const imageUrl = URL.createObjectURL(file);
        uploadedImage.src = imageUrl;

        // Hide upload box and show preview
        fileUploadContainer.classList.add('hidden');
        imagePreview.classList.remove('hidden');
    }
});

// Drag and Drop functionality
fileUploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadContainer.style.borderColor = '#2563eb';
    fileUploadContainer.style.backgroundColor = '#f5f7fa';
});

fileUploadContainer.addEventListener('dragleave', (e) => {
    e.preventDefault();
    fileUploadContainer.style.borderColor = '#d1d5db';
    fileUploadContainer.style.backgroundColor = 'transparent';
});

fileUploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadContainer.style.borderColor = '#d1d5db';
    fileUploadContainer.style.backgroundColor = 'transparent';

    if (e.dataTransfer.files.length > 0) {
        fileUpload.files = e.dataTransfer.files;

        // Trigger change event
        const event = new Event('change');
        fileUpload.dispatchEvent(event);
    }
});

// Submit Button Logic
submitButton.addEventListener('click', () => {
    // Simulate analysis result (use real backend call here if needed)
    analysisResult.classList.remove('hidden');
    resultImage.src = uploadedImage.src; // Placeholder result image
});

// Try Again Button
tryAgainButton.addEventListener('click', () => {
    // Reset
    fileUpload.value = '';
    uploadedImage.src = '';
    resultImage.src = '';
    fileUploadContainer.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    analysisResult.classList.add('hidden');
});
