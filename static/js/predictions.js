function getPrediction(){
    const team = document.getElementById('team-select').value;
    document.getElementById('basketball-animation').style.display = 'block'; // Show the basketball animation
    if(team == ""){
        document.getElementById('prediction-result').innerHTML = `
                <p>Select a team!</p>
        `;
        document.getElementById('basketball-animation').style.display = 'none'; // Hide the basketball animation if no team is selected
        return;
    }
    
    if(team != null){
        document.getElementById('prediction-result').innerHTML = `<p>Prediction for the ${team}'s next game generating... (This may take 15-20 seconds)</p>`;
       

        fetch(`http://127.0.0.1:8000/first_game_predictions?team=${team}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('basketball-animation').style.display = 'none'; // Hide the basketball animation
            document.getElementById('prediction-result').innerHTML = `
                <p>Our model predicts that the ${data.team} will ${data.prediction} their next game against the ${data.opponent} on ${data.date}!</p>
            `;
        })
        .catch(error => {
            document.getElementById('basketball-animation').style.display = 'none'; // Hide the basketball animation on error
            console.error('Error:', error)
        });
    }
}