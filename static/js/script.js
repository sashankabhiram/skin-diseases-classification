document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnRemove = document.getElementById('btn-remove');
    const btnAnalyze = document.getElementById('btn-analyze');
    
    const loadingContainer = document.getElementById('loading-container');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    
    const uploadSection = document.getElementById('upload-section');
    const resultSection = document.getElementById('result-section');
    const btnAnalyzeAnother = document.getElementById('btn-analyze-another');
    
    // Result DOM Elements
    const resultImage = document.getElementById('result-image');
    const predClassName = document.getElementById('pred-class-name');
    const predConfScore = document.getElementById('pred-conf-score');
    const predConfLevel = document.getElementById('pred-conf-level');
    const probBarsContainer = document.getElementById('prob-bars-container');
    
    // Camera DOM Elements
    const btnOpenCamera = document.getElementById('btn-open-camera');
    const cameraContainer = document.getElementById('camera-container');
    const cameraVideo = document.getElementById('camera-video');
    const cameraCanvas = document.getElementById('camera-canvas');
    const btnCloseCamera = document.getElementById('btn-close-camera');
    const btnCapturePhoto = document.getElementById('btn-capture-photo');
    let stream = null;

    // Grad-CAM and Info DOM Elements
    const gradcamImage = document.getElementById('gradcam-image');
    const btnShowOriginal = document.getElementById('btn-show-original');
    const btnShowGradcam = document.getElementById('btn-show-gradcam');
    const infoTitle = document.getElementById('info-title');
    const infoDescription = document.getElementById('info-description');
    const infoUrgency = document.getElementById('info-urgency');
    
    const diseaseInfoMap = {
        'akiec': {
            title: 'Actinic Keratoses',
            description: 'Actinic keratoses (AK) are precancerous skin growths caused by prolonged sun exposure. They often appear as small, rough, or scaly patches on sun-damaged skin.',
            urgency: 'See a dermatologist for evaluation. Left untreated, some can turn into squamous cell carcinoma.'
        },
        'bcc': {
            title: 'Basal Cell Carcinoma',
            description: 'Basal cell carcinoma is the most common form of skin cancer. It often looks like a pearly or waxy bump and rarely spreads, but can cause local tissue damage.',
            urgency: 'Requires medical attention. Highly curable if caught and treated early by a professional.'
        },
        'bkl': {
            title: 'Benign Keratosis-like Lesions',
            description: 'These are non-cancerous growths, often seborrheic keratoses, which appear as warty, brown, or black pasted-on looking spots.',
            urgency: 'Generally harmless. No treatment is needed unless they become irritated or for cosmetic reasons.'
        },
        'df': {
            title: 'Dermatofibroma',
            description: 'A common benign skin growth that often appears as a firm bump on the lower legs. They are thought to be a reaction to minor trauma like bug bites.',
            urgency: 'Harmless and does not require treatment. Consult a doctor if it changes, bleeds, or causes pain.'
        },
        'mel': {
            title: 'Melanoma',
            description: 'Melanoma is the most dangerous type of skin cancer. It develops from the pigment-producing cells and can grow rapidly and spread to other organs.',
            urgency: 'URGENT: Requires immediate evaluation by a dermatologist. Early detection is critical for a high survival rate.'
        },
        'nv': {
            title: 'Melanocytic Nevi (Moles)',
            description: 'Common moles are benign clusters of melanocytes. Almost everyone has them. They are usually uniform in color and have distinct borders.',
            urgency: 'Typically harmless. Monitor for changes in symmetry, border, color, or diameter (ABCD rule).'
        },
        'vasc': {
            title: 'Vascular Lesions',
            description: 'These are relatively common abnormalities of the blood vessels in the skin, such as cherry angiomas or hemangiomas.',
            urgency: 'Usually benign and harmless. Seek medical advice if they bleed frequently or change rapidly.'
        }
    };

    let selectedFile = null;

    // --- Drag and Drop Events ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // --- File Handling ---
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            handleFiles(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (this.files.length) {
            handleFiles(this.files[0]);
        }
    }

    function handleFiles(file) {
        // Reset state
        hideError();
        
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError("Invalid file format. Please upload a JPG or PNG image.");
            return;
        }
        
        // Validate size (16MB max)
        if (file.size > 16 * 1024 * 1024) {
            showError("File is too large. Maximum size is 16MB.");
            return;
        }

        selectedFile = file;
        
        // Show Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadForm.style.display = 'none';
            previewContainer.classList.remove('hidden');
        }
        reader.readAsDataURL(file);
    }

    // --- Camera Logic ---
    btnOpenCamera.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            cameraVideo.srcObject = stream;
            uploadForm.style.display = 'none';
            cameraContainer.classList.remove('hidden');
        } catch (err) {
            console.error("Camera error:", err);
            showError("Could not access camera. Please allow permissions.");
        }
    });

    btnCloseCamera.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        cameraContainer.classList.add('hidden');
        uploadForm.style.display = 'block';
    });

    btnCapturePhoto.addEventListener('click', () => {
        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;
        cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);
        
        cameraCanvas.toBlob((blob) => {
            const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
            handleFiles(file);
            btnCloseCamera.click();
        }, 'image/jpeg', 0.95);
    });

    // --- Button Actions ---
    btnShowOriginal.addEventListener('click', () => {
        resultImage.classList.remove('hidden');
        gradcamImage.classList.add('hidden');
        btnShowOriginal.classList.add('active');
        btnShowGradcam.classList.remove('active');
    });

    btnShowGradcam.addEventListener('click', () => {
        resultImage.classList.add('hidden');
        gradcamImage.classList.remove('hidden');
        btnShowOriginal.classList.remove('active');
        btnShowGradcam.classList.add('active');
    });

    btnRemove.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        uploadForm.style.display = 'block';
        hideError();
    });

    btnAnalyzeAnother.addEventListener('click', () => {
        // Reset everything and go back to upload screen
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        resultSection.classList.add('hidden');
        resultSection.classList.remove('active');
        uploadSection.classList.remove('hidden');
        // Add a tiny delay to allow display:block to apply before animating opacity/transform
        setTimeout(() => uploadSection.classList.add('active'), 50);
        previewContainer.classList.add('hidden');
        uploadForm.style.display = 'block';
        hideError();
    });

    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile) return;
        
        // UI State: Loading
        hideError();
        previewContainer.classList.add('hidden');
        loadingContainer.classList.remove('hidden');
        
        // Create Form Data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            // Send Request
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze image');
            }
            
            // Process Success
            displayResults(data);
            
        } catch (error) {
            console.error('Analysis Error:', error);
            showError(error.message);
            loadingContainer.classList.add('hidden');
            previewContainer.classList.remove('hidden');
        }
    });

    // --- Display Results ---
    function displayResults(data) {
        // Set Image
        resultImage.src = data.image_url;
        
        // Setup Grad-CAM
        if (data.gradcam_url) {
            gradcamImage.src = data.gradcam_url;
            btnShowGradcam.style.display = 'inline-block';
        } else {
            btnShowGradcam.style.display = 'none';
        }
        btnShowOriginal.click(); // Reset toggle to original image view
        
        // Setup Disease Info
        const info = diseaseInfoMap[data.predicted_class_code];
        if (info) {
            infoTitle.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${info.title}`;
            infoDescription.textContent = info.description;
            infoUrgency.textContent = info.urgency;
            
            // Emphasize urgency if Melanoma
            if (data.predicted_class_code === 'mel') {
                infoUrgency.parentElement.style.color = '#b91c1c';
                infoUrgency.parentElement.style.backgroundColor = '#fef2f2';
            } else {
                infoUrgency.parentElement.style.color = '';
                infoUrgency.parentElement.style.backgroundColor = '';
            }
        }
        
        // Set Primary Prediction
        predClassName.textContent = data.predicted_class_name;
        predConfScore.textContent = `${data.confidence_percentage}%`;
        
        // Set Confidence Badge
        predConfLevel.textContent = data.confidence_level;
        predConfLevel.className = 'conf-level'; // reset
        if (data.confidence_percentage >= 85) predConfLevel.classList.add('badge-high');
        else if (data.confidence_percentage >= 50) predConfLevel.classList.add('badge-mod');
        else predConfLevel.classList.add('badge-low');
        
        // Build Probability Bars
        probBarsContainer.innerHTML = '';
        data.all_probabilities.forEach(item => {
            // Determine if this is the predicted class to highlight it
            const isPredicted = item.name === data.predicted_class_name;
            const barStyle = isPredicted ? 'background: var(--accent);' : 'background: var(--accent); opacity: 0.3;';
            const textStyle = isPredicted ? 'color: var(--accent); font-weight: 600;' : '';
            
            const rowHTML = `
                <div class="prob-row">
                    <div class="prob-label">
                        <span style="${textStyle}">${item.name} (${item.code})</span>
                        <span style="${textStyle}">${item.percentage}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: 0%; ${barStyle}" data-width="${item.percentage}%"></div>
                    </div>
                </div>
            `;
            probBarsContainer.insertAdjacentHTML('beforeend', rowHTML);
        });
        
        // Animate bars after a short delay
        setTimeout(() => {
            const bars = document.querySelectorAll('.prob-bar-fill');
            bars.forEach(bar => {
                bar.style.width = bar.getAttribute('data-width');
            });
        }, 100);
        
        // Switch Sections
        loadingContainer.classList.add('hidden');
        uploadSection.classList.add('hidden');
        uploadSection.classList.remove('active');
        resultSection.classList.remove('hidden');
        setTimeout(() => resultSection.classList.add('active'), 50);
    }

    // --- Utils ---
    function showError(msg) {
        errorText.textContent = msg;
        errorMessage.classList.remove('hidden');
    }
    
    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
