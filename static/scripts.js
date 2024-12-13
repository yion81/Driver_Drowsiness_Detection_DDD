// logging js running or not
console.log("a");

function checkDrowsinessStatus() {
    setInterval(function() {
        fetch('/get_drowsy_status')
            .then(response => response.json())
            .then(data => {
                console.log('Drowsy status:', data.is_drowsy);
                const warning = document.getElementById('warning');
                if (data.is_drowsy) {
                    warning.style.display = 'block';
                } else {
                    warning.style.display = 'none';
                }
            })
            .catch(error => console.error('Error:', error));
    }, 1000); // change update in ms
}

checkDrowsinessStatus();