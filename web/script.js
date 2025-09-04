// AI Composition Assistant - JavaScript Frontend

class CompositionAssistant {
    constructor() {
        // Use configuration from config.js
        this.apiUrl = window.APP_CONFIG ? window.APP_CONFIG.API.BASE_URL + '/api' : '/api';
        this.currentImage = null;
        this.analysisResults = null;
        this.config = {
            analysis_depth: 'standard',
            max_suggestions: 5,
            include_technical_metrics: true,
            return_visualizations: false
        };
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input change
        document.getElementById('imageInput').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeImage();
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });
    }

    handleFileSelect(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (50MB limit)
        if (file.size > 50 * 1024 * 1024) {
            this.showError('File size too large. Please select an image smaller than 50MB.');
            return;
        }

        this.currentImage = file;
        this.showImagePreview(file);
    }

    showImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const previewImage = document.getElementById('previewImage');
            previewImage.src = e.target.result;
            previewImage.onload = () => {
                this.showSection('previewSection');
                this.hideSection('uploadArea');
                this.addRuleOfThirdsOverlay();
            };
        };
        reader.readAsDataURL(file);
    }

    addRuleOfThirdsOverlay() {
        const overlay = document.getElementById('imageOverlay');
        const image = document.getElementById('previewImage');
        
        // Clear existing overlay
        overlay.innerHTML = '';
        
        // Create rule of thirds grid
        const grid = document.createElement('div');
        grid.style.position = 'absolute';
        grid.style.top = '0';
        grid.style.left = '0';
        grid.style.width = '100%';
        grid.style.height = '100%';
        grid.style.pointerEvents = 'none';
        
        // Vertical lines
        for (let i = 1; i <= 2; i++) {
            const line = document.createElement('div');
            line.style.position = 'absolute';
            line.style.left = `${(i * 33.33)}%`;
            line.style.top = '0';
            line.style.width = '2px';
            line.style.height = '100%';
            line.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
            line.style.boxShadow = '0 0 2px rgba(0, 0, 0, 0.5)';
            grid.appendChild(line);
        }
        
        // Horizontal lines
        for (let i = 1; i <= 2; i++) {
            const line = document.createElement('div');
            line.style.position = 'absolute';
            line.style.top = `${(i * 33.33)}%`;
            line.style.left = '0';
            line.style.height = '2px';
            line.style.width = '100%';
            line.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
            line.style.boxShadow = '0 0 2px rgba(0, 0, 0, 0.5)';
            grid.appendChild(line);
        }
        
        // Intersection points
        for (let x = 1; x <= 2; x++) {
            for (let y = 1; y <= 2; y++) {
                const point = document.createElement('div');
                point.style.position = 'absolute';
                point.style.left = `${(x * 33.33)}%`;
                point.style.top = `${(y * 33.33)}%`;
                point.style.width = '8px';
                point.style.height = '8px';
                point.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
                point.style.border = '2px solid rgba(0, 0, 0, 0.7)';
                point.style.borderRadius = '50%';
                point.style.transform = 'translate(-50%, -50%)';
                grid.appendChild(point);
            }
        }
        
        overlay.appendChild(grid);
    }

    async analyzeImage() {
        if (!this.currentImage) {
            this.showError('No image selected for analysis.');
            return;
        }

        this.showSection('loadingSection');
        this.hideSection('previewSection');
        this.hideSection('resultsSection');
        this.hideSection('errorSection');

        try {
            const formData = new FormData();
            formData.append('file', this.currentImage);
            formData.append('config', JSON.stringify(this.config));

            const response = await fetch(`${this.apiUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Authorization': 'Bearer demo-token' // For demo purposes
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }

            const results = await response.json();
            this.analysisResults = results;
            this.displayResults(results);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(`Analysis failed: ${error.message}`);
        }
    }

    displayResults(results) {
        this.hideSection('loadingSection');
        
        // Overall Score
        const scoreValue = document.getElementById('scoreValue');
        scoreValue.textContent = results.overall_score.toFixed(3);
        scoreValue.className = `score-value ${this.getScoreClass(results.overall_score)}`;

        // Rule Scores
        this.displayRuleScores(results.rule_scores);

        // Quality Metrics
        this.displayQualityMetrics(results);

        // Suggestions
        this.displaySuggestions(results.suggestions);

        // Initialize detailed analysis tabs
        this.initializeDetailedAnalysis(results);

        this.showSection('resultsSection');
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }

    displayRuleScores(ruleScores) {
        const container = document.getElementById('ruleScores');
        container.innerHTML = '';

        for (const [rule, score] of Object.entries(ruleScores)) {
            const item = document.createElement('div');
            item.className = 'rule-score-item';
            
            const ruleName = this.formatRuleName(rule);
            const scoreClass = this.getScoreClass(score);
            
            item.innerHTML = `
                <div>
                    <span class="rule-name">${ruleName}</span>
                    <div class="score-bar">
                        <div class="score-fill" style="width: ${score * 100}%"></div>
                    </div>
                </div>
                <span class="rule-value ${scoreClass}">${score.toFixed(3)}</span>
            `;
            
            container.appendChild(item);
        }
    }

    displayQualityMetrics(results) {
        const container = document.getElementById('qualityMetrics');
        container.innerHTML = '';

        const metrics = [
            { label: 'Aesthetic Quality', value: results.aesthetic_score },
            { label: 'Technical Quality', value: results.technical_score },
            { label: 'Analysis Confidence', value: results.confidence },
            { label: 'Processing Time', value: results.processing_time, unit: 's', isTime: true }
        ];

        metrics.forEach(metric => {
            const item = document.createElement('div');
            item.className = 'metric-item';
            
            let displayValue = metric.isTime 
                ? `${metric.value.toFixed(3)}${metric.unit}`
                : metric.value.toFixed(3);
            
            const valueClass = metric.isTime ? '' : this.getScoreClass(metric.value);
            
            item.innerHTML = `
                <span class="metric-label">${metric.label}</span>
                <span class="metric-value ${valueClass}">${displayValue}</span>
            `;
            
            container.appendChild(item);
        });
    }

    displaySuggestions(suggestions) {
        const container = document.getElementById('suggestionsList');
        container.innerHTML = '';

        if (!suggestions || suggestions.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: #666;">No specific suggestions available.</p>';
            return;
        }

        suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            
            item.innerHTML = `
                <i class="fas fa-lightbulb suggestion-icon"></i>
                <strong>${index + 1}.</strong> ${suggestion}
            `;
            
            container.appendChild(item);
        });
    }

    initializeDetailedAnalysis(results) {
        // Set default tab content
        this.showTab('ruleOfThirds');
    }

    formatRuleName(rule) {
        return rule.split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    getScoreClass(score) {
        if (score >= 0.8) return 'score-excellent';
        if (score >= 0.6) return 'score-good';
        if (score >= 0.4) return 'score-average';
        return 'score-poor';
    }

    showSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'block';
            section.classList.add('fade-in');
        }
    }

    hideSection(sectionId) {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'none';
            section.classList.remove('fade-in');
        }
    }

    showError(message) {
        this.hideSection('loadingSection');
        this.hideSection('previewSection');
        this.hideSection('resultsSection');
        
        document.getElementById('errorText').textContent = message;
        this.showSection('errorSection');
    }
}

// Tab Management
function showTab(tabName) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Add active class to clicked tab
    event.target.classList.add('active');
    
    const tabContent = document.getElementById('tabContent');
    
    switch(tabName) {
        case 'ruleOfThirds':
            tabContent.innerHTML = `
                <h4><i class="fas fa-th"></i> Rule of Thirds Analysis</h4>
                <p>The rule of thirds divides your image into nine equal sections with two horizontal and two vertical lines. 
                Placing important elements along these lines or at their intersections creates more balanced and visually appealing compositions.</p>
                <div style="margin-top: 1rem;">
                    <strong>What to look for:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li>Main subjects positioned at intersection points</li>
                        <li>Horizon lines aligned with horizontal grid lines</li>
                        <li>Vertical elements (trees, buildings) along vertical lines</li>
                        <li>Balanced distribution of visual weight</li>
                    </ul>
                </div>
            `;
            break;
            
        case 'leadingLines':
            tabContent.innerHTML = `
                <h4><i class="fas fa-arrow-right"></i> Leading Lines Analysis</h4>
                <p>Leading lines guide the viewer's eye through the photograph to the main subject or focal point. 
                They create depth, movement, and visual flow in your composition.</p>
                <div style="margin-top: 1rem;">
                    <strong>Effective leading lines:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li>Roads, paths, or trails</li>
                        <li>Architectural elements (stairs, railings)</li>
                        <li>Natural features (coastlines, rivers)</li>
                        <li>Diagonal, curved, or converging lines</li>
                    </ul>
                </div>
            `;
            break;
            
        case 'symmetry':
            tabContent.innerHTML = `
                <h4><i class="fas fa-balance-scale"></i> Symmetry Analysis</h4>
                <p>Symmetry creates balance and harmony in your composition. It can be vertical, horizontal, or radial, 
                and helps create a sense of order and calm in your images.</p>
                <div style="margin-top: 1rem;">
                    <strong>Types of symmetry:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li><strong>Vertical:</strong> Mirror image along a vertical axis</li>
                        <li><strong>Horizontal:</strong> Reflection along a horizontal line</li>
                        <li><strong>Radial:</strong> Elements radiating from a central point</li>
                        <li><strong>Asymmetrical:</strong> Balanced but not identical elements</li>
                    </ul>
                </div>
            `;
            break;
            
        case 'depth':
            tabContent.innerHTML = `
                <h4><i class="fas fa-layer-group"></i> Depth & Layering Analysis</h4>
                <p>Depth and layering create a three-dimensional feel in your two-dimensional photograph. 
                Multiple layers (foreground, middle ground, background) add visual interest and dimensionality.</p>
                <div style="margin-top: 1rem;">
                    <strong>Creating depth:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li>Overlapping elements</li>
                        <li>Size variation (perspective)</li>
                        <li>Atmospheric perspective (haze, clarity)</li>
                        <li>Depth of field (focus/blur)</li>
                    </ul>
                </div>
            `;
            break;
            
        case 'color':
            tabContent.innerHTML = `
                <h4><i class="fas fa-palette"></i> Color Harmony Analysis</h4>
                <p>Color harmony refers to the pleasing arrangement of colors in your image. 
                Good color combinations create mood, draw attention, and enhance the overall aesthetic appeal.</p>
                <div style="margin-top: 1rem;">
                    <strong>Color schemes:</strong>
                    <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                        <li><strong>Complementary:</strong> Opposite colors on the color wheel</li>
                        <li><strong>Analogous:</strong> Adjacent colors on the color wheel</li>
                        <li><strong>Monochromatic:</strong> Different shades of the same color</li>
                        <li><strong>Triadic:</strong> Three evenly spaced colors</li>
                    </ul>
                </div>
            `;
            break;
    }
}

// Utility Functions
function resetUpload() {
    // Hide all sections except upload
    ['previewSection', 'loadingSection', 'resultsSection', 'errorSection'].forEach(id => {
        document.getElementById(id).style.display = 'none';
    });
    
    // Show upload area
    document.getElementById('uploadArea').style.display = 'block';
    
    // Reset form
    document.getElementById('imageInput').value = '';
    
    // Clear current image
    if (window.compositionAssistant) {
        window.compositionAssistant.currentImage = null;
        window.compositionAssistant.analysisResults = null;
    }
}

function openConfigModal() {
    document.getElementById('configModal').style.display = 'flex';
}

function closeConfigModal() {
    document.getElementById('configModal').style.display = 'none';
}

function applyConfig() {
    if (window.compositionAssistant) {
        window.compositionAssistant.config = {
            analysis_depth: document.getElementById('analysisDepth').value,
            max_suggestions: parseInt(document.getElementById('maxSuggestions').value),
            include_technical_metrics: document.getElementById('includeTechnical').checked,
            return_visualizations: false
        };
    }
    closeConfigModal();
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.compositionAssistant = new CompositionAssistant();
    
    // Show initial upload section
    document.getElementById('uploadArea').style.display = 'block';
    
    console.log('AI Composition Assistant initialized successfully!');
});
