const { load_model, get_prob } = wasm_bindgen;

const canvas = document.getElementById("draw");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "#000";
ctx.fillRect(0, 0, 28, 28);
document.getElementById("clear_btn").onclick = () => {
    ctx.fillRect(0, 0, 28, 28);
    drawn = true;
};

ctx.lineWidth = 2;
document.getElementById("brush_size").oninput = (e) => {
    document.getElementById("brush_size_lb").innerText = `Brush size: ${e.target.value}`;
    ctx.lineWidth = e.target.value;
};

ctx.strokeStyle = "#fff";
document.getElementById("erase_toggle").oninput = (e) => {
    ctx.strokeStyle = e.target.checked ? "#000" : "#fff";
};

let drawn = true;

canvas.onmousemove = (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width * 28;
    const y = (e.clientY - rect.top) / rect.height * 28;

    if (e.buttons & 1) {
        ctx.lineTo(x, y);
        ctx.stroke();
        drawn = true;
    }

    ctx.beginPath();
    ctx.moveTo(x, y);
};

canvas.ontouchstart = (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (e.changedTouches[0].clientX - rect.left) / rect.width * 28;
    const y = (e.changedTouches[0].clientY - rect.top) / rect.height * 28;

    ctx.beginPath();
    ctx.moveTo(x, y);
};

canvas.ontouchmove = (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const x = (e.changedTouches[0].clientX - rect.left) / rect.width * 28;
    const y = (e.changedTouches[0].clientY - rect.top) / rect.height * 28;

    ctx.lineTo(x, y);
    ctx.stroke();
    drawn = true;

    ctx.beginPath();
    ctx.moveTo(x, y);
};

wasm_bindgen().then(() => {
    fetch("model.bin").then((r) => r.arrayBuffer()).then((data) => {
        let model = load_model(new Uint8Array(data));

        setInterval(() => {
            if (!drawn) return;

            drawn = false;

            let p = get_prob(model, ctx.getImageData(0, 0, 28, 28).data);

            let max_v = -Infinity;
            let max_i = 0;
            for (let i = 0; i < 10; i++) {
                document.getElementById("prob" + i).style.width = `${p[i] * 100}%`;

                if (max_v < p[i]) {
                    max_v = p[i];
                    max_i = i;
                }
            }

            document.getElementById("prediction").innerText = `I think it's a ${max_i} (${(max_v * 100).toFixed(2)}% confidence)`
        }, 20);
    });
});
