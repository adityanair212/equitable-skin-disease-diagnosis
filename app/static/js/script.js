const form = document.getElementById('uploadForm'),
      fileInput = document.getElementById('fileInput'),
      previewContainer = document.getElementById('previewContainer'),
      previewImage = document.getElementById('previewImage'),
      spinner = document.getElementById('spinner'),
      resultDiv = document.getElementById('result'),
      predText = document.getElementById('predictionText'),
      confText = document.getElementById('confidenceText'),
      chartCanvas = document.getElementById('chart'),
      gradcamImg = document.getElementById('gradcamImage'),
      historyList = document.getElementById('historyList'),
      backToTopBtn = document.getElementById('backToTopBtn');

let chartInstance;

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    previewContainer.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  previewContainer.classList.add('hidden');
  resultDiv.classList.add('hidden');
  spinner.classList.remove('hidden');

  const data = new FormData(form);
  const res = await fetch('/predict', { method:'POST', body:data });
  const json = await res.json();

  spinner.classList.add('hidden');
  if (json.error) {
    alert(json.error);
    return;
  }

  // Display prediction and softmax-based confidence
  predText.textContent = `Prediction: ${json.prediction}`;
  const predIndex = json.all_labels.indexOf(json.prediction);
  const predConfidence = json.all_confidences[predIndex] * 100;
  confText.textContent = `Confidence: ${predConfidence.toFixed(1)}%`;

  // Chart rendering
  renderChart(json.all_labels, json.all_confidences.map(v => v * 100));

  // Grad-CAM
  gradcamImg.src = json.gradcam_url;
  gradcamImg.classList.remove('hidden');

  resultDiv.classList.remove('hidden');

  // History update
  const li = document.createElement('li');
  li.innerHTML = `<img src="${json.image_url}" /><div><strong>${json.prediction}</strong><br>${predConfidence.toFixed(1)}%</div>`;
  historyList.prepend(li);
});

// Gradient bar chart renderer
function createGradientChart(ctx, labels, data) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 400);
  gradient.addColorStop(0, 'rgba(33, 150, 243, 0.8)');
  gradient.addColorStop(1, 'rgba(76, 175, 80, 0.8)');

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Confidence (%)',
        data: data,
        backgroundColor: gradient,
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          max: 100
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(1)}% confidence`
          }
        }
      }
    }
  });
}

// Chart re-rendering function
function renderChart(labels, values) {
  const ctx = chartCanvas.getContext('2d');
  if (chartInstance) chartInstance.destroy();
  chartInstance = createGradientChart(ctx, labels, values);
}

// Back to Top button functionality
window.onscroll = () => {
  backToTopBtn.style.display =
    (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20)
    ? "block" : "none";
};
backToTopBtn.onclick = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
};
