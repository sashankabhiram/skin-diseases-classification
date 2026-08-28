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

    // --- Button Actions ---
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
        uploadSection.classList.remove('hidden');
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
            // Color based on value
            let color = '#2563eb'; // blue
            if (item.name === data.predicted_class_name) {
                if (item.percentage >= 85) color = '#10b981'; // green
                else if (item.percentage >= 50) color = '#f59e0b'; // yellow
                else color = '#ef4444'; // red
            }
            
            const rowHTML = `
                <div class="prob-row">
                    <div class="prob-label">
                        <span>${item.name} (${item.code})</span>
                        <span>${item.percentage}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: 0%; background-color: ${color};" data-width="${item.percentage}%"></div>
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
        resultSection.classList.remove('hidden');
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
