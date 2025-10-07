const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const browseBtn = document.getElementById('browseBtn');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const preview = document.getElementById('preview');
const resultSection = document.getElementById('resultSection');
const summary = document.getElementById('summary');
const probs = document.getElementById('probs');
const recommendation = document.getElementById('recommendation');
const loading = document.getElementById('loading');
const toasts = document.getElementById('toasts');
const themeToggle = document.getElementById('themeToggle');
const confidencePill = document.getElementById('confidencePill');
const modelSelect = document.getElementById('modelSelect'); // NEW: Get the select element

let selectedFile = null;

// --- NEW: Function to load available models from the API ---
async function loadModels() {
    try {
        const res = await fetch((window.API_BASE || '') + '/models');
        if (!res.ok) throw new Error('Could not load models.');
        const data = await res.json();
        
        modelSelect.innerHTML = ''; // Clear "Loading..." text
        data.models.forEach(model_key => {
            const option = document.createElement('option');
            option.value = model_key;
            // Make names more user-friendly (e.g., 'custom_cnn' becomes 'Custom CNN')
            option.textContent = model_key.replace(/_/g, ' ').replace('cnn', 'CNN').replace(/\b\w/g, l => l.toUpperCase());
            modelSelect.appendChild(option);
        });
    } catch (err) {
        modelSelect.innerHTML = '<option value="">Error loading models</option>';
        toast(err.message, true);
    }
}

// --- CHANGED: Load models on initial page load ---
window.addEventListener('load', () => {
    showLoading(false);
    loadModels(); // Call the function to populate the dropdown
});

function resetResults(){
    resultSection.classList.add('hidden');
    summary.innerHTML = '';
    probs.innerHTML = '';
    recommendation.textContent = '';
    confidencePill.classList.add('hidden');
}

function setPreview(file){
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove('hidden');
    dropzone.querySelector('.placeholder').classList.add('hidden');
    predictBtn.disabled = false;
    clearBtn.disabled = false; // CHANGED: Also enable clear button here
    preview.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
}

dropzone.addEventListener('click', () => fileInput.click());
browseBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });

dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragleave'));
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if(file && file.type.startsWith('image/')){
        selectedFile = file;
        setPreview(file);
        resetResults();
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if(file && file.type.startsWith('image/')){
        selectedFile = file;
        setPreview(file);
        resetResults();
    } else if(file){
        toast('Please select a valid image file', true);
    }
});

clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    preview.classList.add('hidden');
    dropzone.querySelector('.placeholder').classList.remove('hidden');
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    resetResults();
});

// --- CHANGED: predict() now sends the selected model name ---
async function predict(){
    if(!selectedFile){
        toast('Please select an image first');
        return;
    }
    const selectedModel = modelSelect.value;
    if (!selectedModel) {
        toast('Please wait for models to load or select one.', true);
        return;
    }

    predictBtn.disabled = true;
    clearBtn.disabled = true;
    showLoading(true);
    try{
        const form = new FormData();
        form.append('file', selectedFile);
        form.append('model_name', selectedModel); // Add selected model to the form data

        const res = await fetch((window.API_BASE || '') + '/predict', { method:'POST', body: form });
        if(!res.ok){
            const err = await res.json().catch(()=>({detail:'Server error'}));
            throw new Error(err.detail || 'Prediction failed');
        }
        const data = await res.json();
        renderResult(data);
        toast('Analysis complete');
    }catch(err){
        toast(err.message || 'Prediction failed', true);
    }finally{
        showLoading(false);
        predictBtn.disabled = false;
        clearBtn.disabled = false;
    }
}

// --- CHANGED: renderResult() now shows which model was used ---
function renderResult(data){
    resultSection.classList.remove('hidden');
    const pred = data.prediction || {};
    const disease = pred.predicted_disease;
    const conf = Number(pred.confidence || 0);
    const confPct = (conf * 100).toFixed(1) + '%';
    const isHealthy = !!pred.is_healthy;

    // Add a tag to show which model performed the analysis
    const modelUsed = (data.model_used || 'N/A').toUpperCase().replace('_', ' ');
    summary.innerHTML = `Diagnosis: <strong>${disease || '-'}</strong> <span class="model-tag">(via ${modelUsed})</span>`;
    
    confidencePill.textContent = `Confidence: ${confPct}`;
    confidencePill.classList.remove('hidden');

    const probsObj = pred.all_class_probabilities || {};
    probs.innerHTML = '';
    Object.entries(probsObj).sort((a,b)=>b[1]-a[1]).forEach(([cls, p])=>{
        const pct = Math.round(p*100);
        const row = document.createElement('div');
        row.className = 'prob-item' + (!isHealthy && (cls!==disease) ? '' : '');
        row.innerHTML = `
            <span>${cls}</span>
            <div class="bar"><span></span></div>
            <span>${pct}%</span>
        `;
        probs.appendChild(row);
        requestAnimationFrame(()=>{
            const bar = row.querySelector('.bar > span');
            bar.style.width = pct + '%';
        });
    });

    recommendation.textContent = data.recommendation || '';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

predictBtn.addEventListener('click', predict);

function showLoading(show){
    // Use style.display for better consistency
    loading.style.display = show ? 'grid' : 'none';
    loading.setAttribute('aria-hidden', String(!show));
}

function toast(message, isError=false){
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = message;
    if(isError){
        el.style.borderColor = '#7a1f2a';
    }
    toasts.appendChild(el);
    setTimeout(()=>{
        el.style.opacity = '0';
        setTimeout(()=> el.remove(), 300);
    }, 3000);
}

recommendation.addEventListener('click', async () => {
    const text = recommendation.textContent || '';
    if(!text.trim()) return;
    try{
        await navigator.clipboard.writeText(text);
        toast('Recommendation copied to clipboard');
    }catch{
        toast('Copy failed', true);
    }
});

if(themeToggle && themeToggle.style.display !== 'none'){
    themeToggle.addEventListener('click', () => {
        const light = document.documentElement.classList.toggle('light');
        themeToggle.textContent = light ? '☀️' : '🌙';
    });
}