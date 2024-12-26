const { load_model, get_prob } = wasm_bindgen;

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "#000";
ctx.fillRect(0, 0, 28, 28);

ctx.strokeStyle = "#fff";
ctx.lineWidth = 2;

document.getElementById("clear_btn").onclick = () => {
    ctx.fillRect(0, 0, 28, 28);
    drawn = true;
};

let drawn = true;

canvas.ontouchstart = (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (event.changedTouches[0].clientX - rect.left) / rect.width * 28;
    const y = (event.changedTouches[0].clientY - rect.top) / rect.height * 28;

    ctx.beginPath();
    ctx.moveTo(x, y);
};

canvas.ontouchmove = (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (event.changedTouches[0].clientX - rect.left) / rect.width * 28;
    const y = (event.changedTouches[0].clientY - rect.top) / rect.height * 28;

    ctx.lineTo(x, y);
    ctx.stroke();
    drawn = true;

    ctx.beginPath();
    ctx.moveTo(x, y);
};

wasm_bindgen().then(() => {
    fetch("model.bin").then((r) => r.bytes()).then((data) => {
        let model = load_model(data);

        setInterval(() => {
            if (drawn) {
                drawn = false;

                let p = get_prob(model, ctx.getImageData(0, 0, 28, 28).data);

                for (let i = 0; i < 10; i++) {
                    document.getElementById("prob" + i).style.width = `${p[i] * 100}%`;
                }
            }
        }, 20);
    });
});
