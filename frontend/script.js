function predict() {
    const data = {
        gender: Number(document.getElementById("gender").value),
        age: Number(document.getElementById("age").value),

        smoking: Number(document.getElementById("smoking").value),
        yellow_fingers: Number(document.getElementById("yellow_fingers").value),
        anxiety: Number(document.getElementById("anxiety").value),
        peer_pressure: Number(document.getElementById("peer_pressure").value),
        chronic_disease: Number(document.getElementById("chronic_disease").value),
        fatigue: Number(document.getElementById("fatigue").value),
        allergy: Number(document.getElementById("allergy").value),
        wheezing: Number(document.getElementById("wheezing").value),
        alcohol_consuming: Number(document.getElementById("alcohol_consuming").value),
        coughing: Number(document.getElementById("coughing").value),
        shortness_of_breath: Number(document.getElementById("shortness_of_breath").value),
        swallowing_difficulty: Number(document.getElementById("swallowing_difficulty").value),
        chest_pain: Number(document.getElementById("chest_pain").value)
    };

    fetch("https://lung-cancer-prediction-project.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {
        document.getElementById("result").innerText = result.result;
    })
    .catch(err => {
        alert("API Error");
        console.error(err);
    });
}

