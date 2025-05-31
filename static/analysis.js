// DOM Elements
const fileUpload = document.getElementById('fileUpload');
const fileUploadContainer = document.getElementById('fileUploadContainer');
const fileUploadText = document.getElementById('fileUploadText'); // Added to show file name
const imagePreviewDiv = document.getElementById('imagePreview'); // Renamed for clarity
const uploadedImage = document.getElementById('uploadedImage');
const submitButton = document.getElementById('submitButton');
const loadingIndicator = document.getElementById('loadingIndicator'); // Added loading indicator
const analysisResultDiv = document.getElementById('analysisResult'); // Renamed for clarity
const resultImage = document.getElementById('resultImage'); // Changed to use ID from HTML
const tryAgainButton = document.getElementById('tryAgainButton'); // Changed to use ID from HTML
const messageBox = document.getElementById('messageBox'); // Added message box
// Elements for displaying results
const gingivitisStatus = document.getElementById('gingivitisStatus');
const bleedingStatus = document.getElementById('bleedingStatus');
const rednessStatus = document.getElementById('rednessStatus');
const swellingStatus = document.getElementById('swellingStatus');
const detectionsList = document.getElementById('detectionsList');

let selectedFile = null; // To store the file selected by the user

// Function to show a temporary message box
function showMessage(message, type = 'info') {
    messageBox.textContent = message;
    messageBox.style.backgroundColor = type === 'error' ? '#dc2626' : '#333';
    messageBox.style.display = 'block';
    setTimeout(() => {
        messageBox.style.display = 'none';
    }, 3000); // Hide after 3 seconds
}

// Function to handle file selection and preview
function handleFile(file) {
    console.log("handleFile called with file:", file); // Debugging: Check if function is called
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        fileUploadText.textContent = file.name; // Update text to show file name
        submitButton.disabled = false; // Enable submit button

        const reader = new FileReader();
        reader.onload = (e) => {
            console.log("FileReader onload fired. e.target.result length:", e.target.result ? e.target.result.length : 'null'); // Debugging: Check if data URL is generated
            uploadedImage.src = e.target.result; // Set the preview image source
            
            // --- CRUCIAL VISIBILITY TOGGLE ---
            fileUploadContainer.classList.add('hidden'); // HIDE the upload box
            imagePreviewDiv.classList.remove('hidden'); // SHOW the image preview div
            // ----------------------------------
            
            console.log("Visibility updated: fileUploadContainer hidden, imagePreviewDiv shown."); // Debugging: Confirm class changes
        };
        reader.readAsDataURL(file); // Read the file as a data URL
    } else {
        console.log("Invalid file selected."); // Debugging: Confirm file type check
        selectedFile = null;
        fileUploadText.textContent = 'Click or drag and drop your image here';
        submitButton.disabled = true; // Disable submit button
        
        // Ensure upload box is visible if previous file was invalid, hide preview
        fileUploadContainer.classList.remove('hidden'); 
        imagePreviewDiv.classList.add('hidden'); 
        
        uploadedImage.src = ''; // Clear image source
        showMessage("Please upload a valid image file (png, jpg, jpeg).", "error");
    }
}

// Event listener for file input change (when user clicks and selects a file)
fileUpload.addEventListener('change', (event) => {
    handleFile(event.target.files[0]);
});

// Drag and Drop functionality
fileUploadContainer.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileUploadContainer.classList.add('drag-over'); // Use class for styling
});

fileUploadContainer.addEventListener('dragleave', () => {
    fileUploadContainer.classList.remove('drag-over'); // Remove class for styling
});

fileUploadContainer.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadContainer.classList.remove('drag-over'); // Remove class for styling

    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]); // Handle the dropped file
    }
});

// Submit Button Logic
submitButton.addEventListener('click', async () => {
    if (!selectedFile) {
        showMessage("Please upload an image first.", "error");
        return;
    }

    // Show loading indicator and disable button
    loadingIndicator.classList.remove('hidden');
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        console.log("Response OK?", response.ok);
        console.log("Response data:", data);

        if (response.ok) {
            // Hide upload section and show analysis result section
            document.getElementById('uploadSection').classList.add('hidden');
            analysisResultDiv.classList.remove('hidden');

            // CRUCIAL FIX: Set the result image source to the base64 image from the backend
            if (data.image) {
                resultImage.src = `data:image/jpeg;base64,${data.image}`;
            } else {
                showMessage("No image with bounding boxes received from server.", "error");
                resultImage.src = ''; // Clear image if no result image
            }

            // Update symptom statuses
            gingivitisStatus.textContent = data.gingivitis ? 'Detected' : 'Not Detected';
            gingivitisStatus.classList.remove('detected', 'not-detected'); // Clear previous classes
            gingivitisStatus.classList.add(data.gingivitis ? 'detected' : 'not-detected');

            bleedingStatus.textContent = data.bleeding ? 'Detected' : 'Not Detected';
            bleedingStatus.classList.remove('detected', 'not-detected');
            bleedingStatus.classList.add(data.bleeding ? 'detected' : 'not-detected');

            rednessStatus.textContent = data.redness ? 'Detected' : 'Not Detected';
            rednessStatus.classList.remove('detected', 'not-detected');
            rednessStatus.classList.add(data.redness ? 'detected' : 'not-detected');

            swellingStatus.textContent = data.swelling ? 'Detected' : 'Not Detected';
            swellingStatus.classList.remove('detected', 'not-detected');
            swellingStatus.classList.add(data.swelling ? 'detected' : 'not-detected');

            // Display detected symptoms list
            detectionsList.innerHTML = ''; // Clear previous detections
            if (data.detections && data.detections.length > 0) {
                data.detections.forEach(detection => {
                    const listItem = document.createElement('li');
                    listItem.textContent = `${detection.class} (Confidence: ${(detection.confidence * 100).toFixed(1)}%)`;
                    detectionsList.appendChild(listItem);
                });
            } else {
                const listItem = document.createElement('li');
                listItem.textContent = 'No specific symptoms detected with current thresholds.';
                detectionsList.appendChild(listItem);
            }

        } else {
            showMessage("Error: " + (data.error || "Unknown error during analysis."), "error");
            // Reset results display in case of error
            gingivitisStatus.textContent = 'Error';
            gingivitisStatus.className = 'status-value detected'; // Indicate error with red
            bleedingStatus.textContent = 'Error';
            rednessStatus.textContent = 'Error';
            swellingStatus.textContent = 'Error';
            detectionsList.innerHTML = '<li>An error occurred during analysis.</li>';
            resultImage.src = ''; // Clear result image on error
        }

    } catch (err) {
        console.error("Upload failed:", err);
        showMessage("An error occurred while uploading the image.", "error");
        // Reset results display in case of error
        gingivitisStatus.textContent = 'Error';
        gingivitisStatus.className = 'status-value detected'; // Indicate error with red
        bleedingStatus.textContent = 'Error';
        rednessStatus.textContent = 'Error';
        swellingStatus.textContent = 'Error';
        detectionsList.innerHTML = '<li>An error occurred during analysis.</li>';
        resultImage.src = ''; // Clear result image on error
    } finally {
        // Hide loading indicator and re-enable button
        loadingIndicator.classList.add('hidden');
        submitButton.disabled = false;
    }
});

// Try Again Button
tryAgainButton.addEventListener('click', () => {
    // Reset
    selectedFile = null; // Clear selected file
    fileUpload.value = ''; // Clear the file input
    fileUploadText.textContent = 'Click or drag and drop your image here'; // Reset text
    uploadedImage.src = ''; // Clear preview image
    imagePreviewDiv.classList.add('hidden'); // Hide preview div
    submitButton.disabled = true; // Disable submit button
    
    // Show upload section and hide analysis result section
    document.getElementById('uploadSection').classList.remove('hidden');
    analysisResultDiv.classList.add('hidden');

    // Ensure the fileUploadContainer is visible again
    fileUploadContainer.classList.remove('hidden'); 

    // Reset analysis results display
    resultImage.src = '';
    gingivitisStatus.textContent = 'N/A';
    gingivitisStatus.classList.remove('detected', 'not-detected');
    bleedingStatus.textContent = 'N/A';
    bleedingStatus.classList.remove('detected', 'not-detected');
    rednessStatus.textContent = 'N/A';
    rednessStatus.classList.remove('detected', 'not-detected');
    swellingStatus.textContent = 'N/A';
    swellingStatus.classList.remove('detected', 'not-detected');
    detectionsList.innerHTML = '<li>No detections yet.</li>';
});
