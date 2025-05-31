// analysis.js
// DOM Elements
const fileUpload = document.getElementById('fileUpload');
const fileUploadContainer = document.getElementById('fileUploadContainer');
const imagePreview = document.getElementById('imagePreview');
const uploadedImage = document.getElementById('uploadedImage');
const submitButton = document.getElementById('submitButton');
const analysisResult = document.getElementById('analysisResult');
const resultImage = document.querySelector('.result-image');
const tryAgainButton = document.querySelector('.try-again-button');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

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
submitButton.addEventListener('click', async () => {
    const file = fileUpload.files[0];
    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        console.log("Response OK?", response.ok);
        console.log("Response data:", data);

        if (response.ok) {
            // Show result section and set image (triggers .onload)
            analysisResult.classList.remove('hidden');
            resultImage.src = uploadedImage.src;

            // When the image is fully loaded, draw on the canvas
            uploadedImage.onload = () => {
                canvas.width = uploadedImage.width;
                canvas.height = uploadedImage.height;

                // Clear canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw detections if any
                if (data.detections && Array.isArray(data.detections)) {
                    ctx.lineWidth = 2;
                    ctx.font = "16px Arial";

                    data.detections.forEach(det => {
                        const [x1, y1, x2, y2] = det.bbox;
                        ctx.strokeStyle = "red";
                        ctx.fillStyle = "red";
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                        ctx.fillText(`${det.class} (${(det.confidence * 100).toFixed(1)}%)`, x1, y1 - 5);
                        console.log("Drawing box for:", det);
                    });
                }
            };

            // Append the results dynamically
            const resultContainer = document.querySelector('.result-container');

            // Clear previous summary if it exists
            const oldSummary = resultContainer.querySelector('.summary');
            if (oldSummary) {
                oldSummary.remove();
            }

            let summaryHTML = `<div class="summary">
                <p><strong>Result:</strong> ${
                    data.gingivitis ? "Gingivitis Detected" : "No Gingivitis Detected"
                }</p>
                <ul>
                    <li>Bleeding: ${data.bleeding ? "✅" : "❌"}</li>
                    <li>Redness: ${data.redness ? "✅" : "❌"}</li>
                    <li>Swelling: ${data.swelling ? "✅" : "❌"}</li>
                </ul>
            </div>`;

            resultContainer.insertAdjacentHTML('beforeend', summaryHTML);
        } else {
            alert("Error: " + (data.error || "Unknown error during analysis."));
        }

    } catch (err) {
        console.error("Upload failed:", err);
        alert("An error occurred while uploading the image.");
    }
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
