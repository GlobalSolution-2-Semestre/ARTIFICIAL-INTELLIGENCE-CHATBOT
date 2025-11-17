document.querySelectorAll("input[type='number']").forEach(input => {
    input.addEventListener("input", () => {
        if (input.value > 10) input.value = 10;
        if (input.value < 0) input.value = 0;
    });
});
