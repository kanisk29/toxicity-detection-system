// ---------------- THREE.JS BACKGROUND ----------------
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);

const renderer = new THREE.WebGLRenderer({
  canvas: document.querySelector("#bg"),
  antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

camera.position.z = 30;

// Geometry (more premium than torus)
const geometry = new THREE.IcosahedronGeometry(10, 1);
const material = new THREE.MeshStandardMaterial({
  color: 0x00f5a0,
  wireframe: true
});

const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

// Lighting
const light = new THREE.PointLight(0xffffff, 1);
light.position.set(20, 20, 20);
scene.add(light);

// Animation loop
function animate() {
  requestAnimationFrame(animate);

  mesh.rotation.x += 0.003;
  mesh.rotation.y += 0.005;

  renderer.render(scene, camera);
}
animate();

// Resize fix
window.addEventListener("resize", () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});


// ---------------- API CALL ----------------
const button = document.getElementById("analyzeBtn");
button.addEventListener("click", predict);

async function predict() {
  const text = document.getElementById("inputText").value;
  const resultsDiv = document.getElementById("results");
  const loader = document.getElementById("loader");

  if (!text.trim()) {
    alert("Enter some text");
    return;
  }

  resultsDiv.innerHTML = "";
  loader.classList.remove("hidden");
  button.disabled = true;

  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    loader.classList.add("hidden");
    button.disabled = false;

    renderResults(data);

  } catch (error) {
    loader.classList.add("hidden");
    button.disabled = false;

    resultsDiv.innerHTML =
      "<p style='color:red;'>Backend not reachable</p>";
  }
}


// ---------------- RENDER RESULTS ----------------
function renderResults(data) {
  const container = document.getElementById("results");
  container.innerHTML = "";

  const results = data.results;

  Object.keys(results).forEach(label => {
    const prob = results[label].confidence;
    const isToxic = results[label].prediction === 1;

    const div = document.createElement("div");
    div.className = "result";

    div.innerHTML = `
      <div>
        ${label} 
        <span style="float:right; color:${isToxic ? "#ff4d4d" : "#00f5a0"}">
          ${(prob * 100).toFixed(1)}%
        </span>
      </div>
      <div class="bar">
        <div class="fill"></div>
      </div>
    `;

    container.appendChild(div);

    // Animate bar
    setTimeout(() => {
      div.querySelector(".fill").style.width = (prob * 100) + "%";

      if (isToxic) {
        div.querySelector(".fill").style.background = "#ff4d4d";
      }
    }, 100);
  });
}