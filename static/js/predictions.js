function getPrediction(){
    const team = document.getElementById('team-select').value;
    if(team != null){
        document.getElementById('prediction-result').innerHTML = `<p>Prediction for the ${team}'s next game generating... (This may take 10-15 seconds)</p>`;
        fetch(`http://127.0.0.1:8000/first_game_predictions?team=${team}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('prediction-result').innerHTML = `
                <p>Our model predicts that the ${data.team} will ${data.prediction} their next game against the ${data.opponent} on ${data.date}!</p>
            `;
        })
        .catch(error => console.error('Error:', error));
    }
}