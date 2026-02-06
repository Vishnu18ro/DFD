const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const previewContainer = document.getElementById('preview-container');
const dropContent = document.querySelector('.drop-content');
const removeBtn = document.getElementById('remove-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const spinner = document.querySelector('.spinner');
const btnText = analyzeBtn.querySelector('span');
const resultContainer = document.getElementById('result-container');
const scoreStroke = document.getElementById('score-stroke');
const scoreValue = document.getElementById('score-value');
const predictionLabel = document.getElementById('prediction-label');

let currentFile = null;

// Paste Event
document.addEventListener('paste', function (e) {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
            const blob = items[i].getAsFile();
            // Create a file from the blob with a default name
            const file = new File([blob], "pasted_image.png", { type: blob.type });
            handleFiles([file]);
            break;
        }
    }
});

// Drag & Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    dropZone.classList.add('dragover');
}

function unhighlight(e) {
    dropZone.classList.remove('dragover');
}

dropZone.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

fileInput.addEventListener('change', function () {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        currentFile = files[0];
        if (currentFile.type.startsWith('image/')) {
            showPreview(currentFile);
            analyzeBtn.disabled = false;
            resultContainer.classList.add('hidden'); // Hide old results
        } else {
            alert('Please upload an image file.');
        }
    }
}

function showPreview(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = function () {
        imagePreview.src = reader.result;
        dropContent.style.display = 'none';
        previewContainer.classList.remove('hidden');
    }
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent opening file dialog
    resetUI();
});

function resetUI() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewContainer.classList.add('hidden');
    dropContent.style.display = 'flex';
    analyzeBtn.disabled = true;
    resultContainer.classList.add('hidden');
}

// Analyze
analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Loading State
    analyzeBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    spinner.classList.remove('hidden');
    resultContainer.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showResult(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis.');
    } finally {
        analyzeBtn.disabled = false;
        btnText.textContent = 'Analyze Image';
        spinner.classList.add('hidden');
    }
});

function showResult(data) {
    resultContainer.classList.remove('hidden');

    // Set text
    predictionLabel.textContent = data.label;

    // Set color
    if (data.label.includes("AI")) {
        predictionLabel.className = 'label ai';
        scoreStroke.style.stroke = 'var(--ai-color)';
    } else {
        predictionLabel.className = 'label real';
        scoreStroke.style.stroke = 'var(--real-color)';
    }

    // Animate Score
    const score = Math.round(data.score);
    animateScore(score);
}

function animateScore(target) {
    let current = 0;
    const duration = 1000;
    const startTime = performance.now();

    function update(time) {
        const elapsed = time - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing (easeOutQuart)
        const ease = 1 - Math.pow(1 - progress, 4);

        current = Math.floor(target * ease);
        scoreValue.textContent = current;

        // Update stroke dasharray 
        // 100 is the circumference (roughly, as set in dasharray logic but here we use percent)
        scoreStroke.style.strokeDasharray = `${current}, 100`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// Modal Logic
const modal = document.getElementById('example-modal');
const btn = document.getElementById('examples-btn');
const closeBtn = document.querySelector('.modal-close');

btn.onclick = function () {
    modal.classList.add('active');
}

closeBtn.onclick = function () {
    modal.classList.remove('active');
}

window.onclick = function (event) {
    if (event.target == modal) {
        modal.classList.remove('active');
    }
}

// Example Gallery Logic
async function loadExample(url) {
    // Close modal if open
    if (modal) modal.classList.remove('active');

    try {
        const response = await fetch(url);
        const blob = await response.blob();

        // Extract filename from URL
        const filename = url.split('/').pop();

        // Create file object
        const file = new File([blob], filename, { type: blob.type });

        // Reuse handleFiles logic
        handleFiles([file]);

        // Scroll up to results
        window.scrollTo({ top: 0, behavior: 'smooth' });

    } catch (e) {
        console.error("Error loading example:", e);
        alert("Could not load example image.");
    }
}

// Particle System
const canvas = document.getElementById('particles');
const ctx = canvas.getContext('2d');
let particles = [];

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

class Particle {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.size = Math.random() * 2 + 1;
        this.speedX = Math.random() * 2 - 1;
        this.speedY = Math.random() * 2 - 1;
        this.color = `rgba(255, 255, 255, ${Math.random() * 0.3 + 0.1})`;
    }

    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        if (this.x > canvas.width || this.x < 0) this.speedX *= -1;
        if (this.y > canvas.height || this.y < 0) this.speedY *= -1;

        // Mouse interaction
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 100) {
            this.x -= dx * 0.02;
            this.y -= dy * 0.02;
        }
    }

    draw() {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
    }
}

function initParticles() {
    for (let i = 0; i < 50; i++) {
        particles.push(new Particle());
    }
}

const mouse = { x: null, y: null };
window.addEventListener('mousemove', function (e) {
    mouse.x = e.x;
    mouse.y = e.y;
});

function animateParticles() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < particles.length; i++) {
        particles[i].update();
        particles[i].draw();

        // Connect particles
        for (let j = i; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < 100) {
                ctx.beginPath();
                ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance / 100 * 0.1})`;
                ctx.lineWidth = 0.5;
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.stroke();
            }
        }
    }
    requestAnimationFrame(animateParticles);
}

initParticles();
animateParticles();
