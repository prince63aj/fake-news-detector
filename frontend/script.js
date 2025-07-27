async function checkNews() {
  const text = document.getElementById("newsText").value.trim();

  if (!text) {
    alert("Please enter some news content.");
    return;
  }

  const response = await fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: text })
  });

  const resultDiv = document.getElementById("result");

  if (response.ok) {
    const data = await response.json();
    resultDiv.innerText = `🧠 Prediction: ${data.prediction}`;
    resultDiv.style.color = data.prediction === "Fake News" ? "red" : "green";
  } else {
    resultDiv.innerText = "⚠️ Error: Unable to connect to the backend.";
    resultDiv.style.color = "gray";
  }
}
