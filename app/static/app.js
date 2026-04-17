document.addEventListener('DOMContentLoaded', () => {
    // Load initial stats
    fetchStats();

    const analyzeBtn = document.getElementById('analyze-btn');
    const inputField = document.getElementById('query-input');

    analyzeBtn.addEventListener('click', handleAnalysis);
    inputField.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleAnalysis();
    });

    // Wire up inactive buttons for demo purposes
    document.querySelectorAll('.inactive-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const featureName = e.currentTarget.getAttribute('data-feat') || 'This feature';
            showToast(`${featureName} is disabled in the Hackathon Demo.`);
        });
    });
});

function showToast(message) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<i data-feather="info"></i> <span>${message}</span>`;
    container.appendChild(toast);
    feather.replace();
    
    // Remove after animation finishes (3.4s)
    setTimeout(() => {
        if(container.contains(toast)) {
            container.removeChild(toast);
        }
    }, 3500);
}

async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        document.getElementById('dict-size').textContent = `${data.dictionary_size} Compounds`;
        document.getElementById('model-name').textContent = data.model_name;
    } catch (e) {
        console.error("Failed to load generic stats", e);
    }
}

function resetUI() {
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    
    const card = document.getElementById('verdict-card');
    card.className = "verdict-card glass-panel"; // reset classes
    
    document.getElementById('resolution-badge').classList.add('hidden');
    document.getElementById('probability-container').classList.add('hidden');
    document.getElementById('priority-alert').classList.add('hidden');
    
    // Set loading state
    document.getElementById('verdict-title').textContent = "Processing...";
    document.getElementById('verdict-subtitle').textContent = "Querying Neural Network and Registries...";
    document.getElementById('verdict-icon').setAttribute('data-feather', 'loader');
    
    // Animate search button
    const btnBox = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    btnBox.classList.add('hidden');
    loader.classList.remove('hidden');
    
    // Replace feather icon
    feather.replace();
}

async function handleAnalysis() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    resetUI();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        if(!response.ok) throw new Error(data.detail || "API Error");
        
        renderResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        const btnBox = document.querySelector('.btn-text');
        const loader = document.querySelector('.loader');
        btnBox.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function renderResults(data) {
    const card = document.getElementById('verdict-card');
    const icon = document.getElementById('verdict-icon');
    const title = document.getElementById('verdict-title');
    const subtitle = document.getElementById('verdict-subtitle');
    
    // Render canonical SMILES
    document.getElementById('meta-smiles').textContent = data.smiles || 'N/A';
    
    // Show PubChem resolution badge if needed
    if (data.resolved_via && data.resolved_via.includes('pubchem')) {
        const badge = document.getElementById('resolution-badge');
        badge.innerHTML = `<i data-feather="check-circle"></i> Resolved "${data.resolved_name}" via PubChem API`;
        badge.classList.remove('hidden');
    }

    if (data.is_priority_toxin) {
        // Priority Toxin matched
        card.classList.add('toxic');
        title.textContent = "CRITICAL HAZARD";
        subtitle.textContent = "AOT Dictionary Bypass Triggered.";
        icon.setAttribute('data-feather', 'alert-triangle');
        
        // Show priority alert box
        const alertBox = document.getElementById('priority-alert');
        alertBox.classList.remove('hidden');
        document.getElementById('reg-name').textContent = data.name;
        document.getElementById('reg-source').textContent = data.source;
        document.getElementById('reg-hazard').textContent = data.hazard_class;
        
        // Hide Probability Container
        document.getElementById('probability-container').classList.add('hidden');
        
        document.getElementById('meta-domain').textContent = "Not Applicable (Bypass)";

    } else {
        // ML Inference
        const probContainer = document.getElementById('probability-container');
        probContainer.classList.remove('hidden');
        
        // Fill Progress Bar
        const pct = (data.probability * 100).toFixed(1);
        document.getElementById('prob-percentage').innerHTML = `${pct}% <span style="font-size:0.7rem; color:var(--text-muted); font-weight:500;">TOLERANCE</span>`;
        setTimeout(() => {
            const fill = document.getElementById('prob-fill');
            fill.style.width = `${pct}%`;
        }, 100);

        // Update Card Styles based on verdict
        if (data.verdict === 'SAFE') {
            card.classList.add('safe');
            title.textContent = "Likely Safe";
            subtitle.textContent = "Predicted toxicity remains below critical thresholds.";
            icon.setAttribute('data-feather', 'shield-check');
        } else if (data.verdict === 'CRITICAL HAZARD') {
            card.classList.add('toxic');
            title.textContent = "Toxic Signal Detected";
            subtitle.textContent = "Model identifies significant molecular hazards.";
            icon.setAttribute('data-feather', 'alert-circle');
        } else {
            card.classList.add('uncertain');
            title.textContent = "Uncertain / Borderline";
            subtitle.textContent = "A manual secondary review is recommended.";
            icon.setAttribute('data-feather', 'info');
        }
        
        // Applicability Domain
        const domNode = document.getElementById('meta-domain');
        if (data.in_envelope) {
            domNode.innerHTML = `<span style="color: var(--accent-green)">Within Training Distribution</span>`;
        } else {
            domNode.innerHTML = `<span style="color: var(--accent-red)">Out of Distribution (OOD)</span>`;
            // Add a small warning to the verdict
            subtitle.textContent += " Warning: Extrapolation detected.";
        }
    }
    
    // Set Confidence
    const conf = (data.confidence * 100).toFixed(0);
    const confBadge = document.getElementById('confidence-badge');
    confBadge.textContent = `${conf}% Confidence`;
    
    if (data.confidence < 0.6) {
        confBadge.style.color = '#FFB340';
        confBadge.style.borderColor = '#FFB340';
    } else {
        confBadge.style.color = '#fff';
        confBadge.style.borderColor = 'var(--border-color)';
    }

    feather.replace();
}

function showError(msg) {
    const card = document.getElementById('verdict-card');
    card.classList.add('uncertain');
    document.getElementById('verdict-title').textContent = "Prediction Error";
    document.getElementById('verdict-subtitle').textContent = msg;
    document.getElementById('verdict-icon').setAttribute('data-feather', 'x-octagon');
    document.getElementById('resolution-badge').classList.add('hidden');
    feather.replace();
}
