const historyDiv = document.getElementById("history");
const toggle = document.getElementById("themeToggle");

// sound
const clickSound = new Audio("https://www.soundjay.com/buttons/sounds/button-16.mp3");

function addToHistory(text) {
  const item = document.createElement("div");
  item.innerText = text.slice(0, 40) + "...";
  item.onclick = () => {
    document.getElementById("text").value = text;
  };
  historyDiv.prepend(item);
}

// theme toggle
toggle.onclick = () => {
  document.body.classList.toggle("light");

  toggle.innerText = document.body.classList.contains("light")
    ? "☀️"
    : "🌙";
};

async function predict() {
  clickSound.play();

  const text = document.getElementById("text").value;
  const model = document.getElementById("model").value;
  const resultDiv = document.getElementById("result");

  if (!text.trim()) {
    resultDiv.innerHTML = "<p>Enter text first</p>";
    return;
  }

  resultDiv.innerHTML = `<div class="loader"></div>`;

  try {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text, model })
    });

    const data = await res.json();

    addToHistory(text);

    const modelName = data.model_used.toUpperCase();
    let output = `<h3 class="typing">Model: ${modelName}</h3>`;
    output += `<div>${data.is_toxic ? "This comment has been flagged as TOXIC!!!" : "This comment is SAFE"}</div>`;

    data.labels.forEach((item, i) => {
      const percent = (item.score * 100).toFixed(1);
      const cls = item.flagged ? "danger" : "safe";

      output += `
        <div class="label" style="animation-delay:${i * 0.1}s">
          <div>${item.label}</div>
          <div class="bar">
            <div class="fill ${cls}" style="width:${percent}%"></div>
          </div>
        </div>
      `;
    });

    resultDiv.innerHTML = output;

    // animate bars
    setTimeout(() => {
      document.querySelectorAll(".fill").forEach(el => {
        el.style.width = el.style.width;
      });
    }, 100);

  } catch (err) {
    resultDiv.innerHTML = `<p>Error: ${err.message}</p>`;
  }
}