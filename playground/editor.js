
import statik from '../statik.js';

document.getElementById("run").addEventListener("click", () => {
  const code = document.getElementById("editor").value;
  let result = "";
  try {
    result = eval(code);
  } catch (e) {
    result = "Error: " + e.message;
  }
  document.getElementById("output").textContent = result;
});
