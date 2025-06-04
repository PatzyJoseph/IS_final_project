console.log("Script loaded");

// DOM Elements
const fileUpload = document.getElementById('fileUpload');
const fileUploadContainer = document.getElementById('fileUploadContainer');
const fileUploadText = document.getElementById('fileUploadText');
const imagePreviewDiv = document.getElementById('imagePreview');
const uploadedImage = document.getElementById('uploadedImage');
const submitButton = document.getElementById('submitButton');
const loadingIndicator = document.getElementById('loadingIndicator');
const analysisResultDiv = document.getElementById('analysisResult');
const resultImage = document.getElementById('resultImage');
const tryAgainButton = document.getElementById('tryAgainButton');
const messageBox = document.getElementById('messageBox');
const gingivitisStatus = document.getElementById('gingivitisStatus');
const bleedingStatus = document.getElementById('bleedingStatus');
const rednessStatus = document.getElementById('rednessStatus');
const swellingStatus = document.getElementById('swellingStatus');
const detectionsList = document.getElementById('detectionsList');

let selectedFile = null;

// Show message
function showMessage(message, type = 'info') {
    messageBox.textContent = message;
    messageBox.style.backgroundColor = type === 'error' ? '#dc2626' : '#333';
    messageBox.style.display = 'block';
    setTimeout(() => {
        messageBox.style.display = 'none';
    }, 3000);
}

// Load recommendations JSON
async function loadRecommendations() {
    try {
        const response = await fetch("static/recommendations.json");
        if (!response.ok) throw new Error('Failed to load recommendations.json');
        return await response.json();
    } catch (error) {
        console.error("Error loading recommendations:", error);
        return {};
    }
}

// Display recommendations
function displayRecommendations(detectedSymptoms, recommendations) {
    const section = document.getElementById("recommendationsSection");
    const messageBox = document.getElementById("recommendationContent");
    messageBox.innerHTML = "";
    section.style.display = "block";

    detectedSymptoms.forEach(symptom => {
        const data = recommendations[symptom];
        if (data) {
            const div = document.createElement("div");
            div.className = "recommendation-section";
            div.innerHTML = `
                <h4>${symptom.charAt(0).toUpperCase() + symptom.slice(1)} Recommendation</h4>
                <p><strong>Message:</strong> ${data.primary_message}</p>
                <p><strong>Immediate Recommendations:</strong> ${data.immediate_recommendations}</p>
                <p><strong>Watch for:</strong> ${data.watch_for_additional_symptoms}</p>
                <p><strong>Possible Causes:</strong> ${data.possible_causes}</p>
                <p><strong>Consultation Advice:</strong> ${data.professional_consultation}</p>
                <p><strong>Prevention Tips:</strong> ${data.prevention}</p>
                <hr>
            `;
            messageBox.appendChild(div);
        }
    });
}

// Handle file
function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        fileUploadText.textContent = file.name;
        submitButton.disabled = false;

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
            fileUploadContainer.classList.add('hidden');
            imagePreviewDiv.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        selectedFile = null;
        fileUploadText.textContent = 'Click or drag and drop your image here';
        submitButton.disabled = true;
        uploadedImage.src = '';
        fileUploadContainer.classList.remove('hidden');
        imagePreviewDiv.classList.add('hidden');
        showMessage("Please upload a valid image file.", "error");
    }
}

// File input event
fileUpload.addEventListener('change', (e) => handleFile(e.target.files[0]));

// Drag and drop
fileUploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadContainer.classList.add('drag-over');
});
fileUploadContainer.addEventListener('dragleave', () => {
    fileUploadContainer.classList.remove('drag-over');
});
fileUploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadContainer.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

// Submit Button
submitButton.addEventListener('click', async () => {
    if (!selectedFile) {
        showMessage("Please upload an image first.", "error");
        return;
    }

    loadingIndicator.classList.remove('hidden');
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (response.ok) {
            document.getElementById('uploadSection').classList.add('hidden');
            analysisResultDiv.classList.remove('hidden');

            if (data.image) {
                resultImage.src = `data:image/jpeg;base64,${data.image}`;
            }

            gingivitisStatus.textContent = data.gingivitis ? 'Detected' : 'Not Detected';
            gingivitisStatus.className = 'status-value ' + (data.gingivitis ? 'detected' : 'not-detected');

            bleedingStatus.textContent = data.bleeding ? 'Detected' : 'Not Detected';
            bleedingStatus.className = 'status-value ' + (data.bleeding ? 'detected' : 'not-detected');

            rednessStatus.textContent = data.redness ? 'Detected' : 'Not Detected';
            rednessStatus.className = 'status-value ' + (data.redness ? 'detected' : 'not-detected');

            swellingStatus.textContent = data.swelling ? 'Detected' : 'Not Detected';
            swellingStatus.className = 'status-value ' + (data.swelling ? 'detected' : 'not-detected');

            detectionsList.innerHTML = '';
            if (data.detections?.length) {
                data.detections.forEach(detection => {
                    const li = document.createElement('li');
                    li.textContent = `${detection.class} (Confidence: ${(detection.confidence * 100).toFixed(1)}%)`;
                    detectionsList.appendChild(li);
                });
            } else {
                detectionsList.innerHTML = '<li>No specific symptoms detected with current thresholds.</li>';
            }

            // ✅ Display Recommendations based on detections
            const detectedSymptoms = data.symptom_combinations || [];

            if (detectedSymptoms.length > 0) {
                // Sort to prioritize most specific combination
                detectedSymptoms.sort((a, b) => a.split('_').length - b.split('_').length);
                const mostSpecificSymptom = detectedSymptoms[detectedSymptoms.length - 1];
                const recommendations = await loadRecommendations();
                displayRecommendations([mostSpecificSymptom], recommendations);
            } else {
                const recommendations = await loadRecommendations();
                displayRecommendations([], recommendations);
            }

        } else {
            showMessage("Error: " + (data.error || "Unknown error during analysis."), "error");
        }

    } catch (err) {
        console.error("Upload failed:", err);
        showMessage("An error occurred while uploading the image.", "error");
    } finally {
        loadingIndicator.classList.add('hidden');
        submitButton.disabled = false;
    }
});

// Try Again Button
tryAgainButton.addEventListener('click', () => {
    selectedFile = null;
    fileUpload.value = '';
    fileUploadText.textContent = 'Click or drag and drop your image here';
    uploadedImage.src = '';
    imagePreviewDiv.classList.add('hidden');
    submitButton.disabled = true;

    document.getElementById('uploadSection').classList.remove('hidden');
    analysisResultDiv.classList.add('hidden');
    fileUploadContainer.classList.remove('hidden');

    resultImage.src = '';
    gingivitisStatus.textContent = 'N/A';
    bleedingStatus.textContent = 'N/A';
    rednessStatus.textContent = 'N/A';
    swellingStatus.textContent = 'N/A';
    gingivitisStatus.classList.remove('detected', 'not-detected');
    bleedingStatus.classList.remove('detected', 'not-detected');
    rednessStatus.classList.remove('detected', 'not-detected');
    swellingStatus.classList.remove('detected', 'not-detected');
    detectionsList.innerHTML = '<li>No detections yet.</li>';

    document.getElementById("recommendationsSection").style.display = "none";
    document.getElementById("recommendationContent").innerHTML = '';
});