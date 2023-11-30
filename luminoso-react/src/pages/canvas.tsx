import {useEffect, useState} from "react";

function Canvas() {
    let painting = false;

    useEffect(() => {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";

        ctx.lineWidth = 6;
        ctx.lineCap = "round";
        ctx.strokeStyle = "#ffffff";
        canvas.width = 300;
        canvas.height = 300;

        canvas.addEventListener("mouseup", () => {
            painting = false;
        });
        canvas.addEventListener("mousedown", () => {
            painting = true;
            ctx.lineWidth = 6;
        });
        canvas.addEventListener("mousemove", (event) => {
            const x = event.offsetX;
            const y = event.offsetY;

            if(!painting) {
                ctx.beginPath();
                ctx.moveTo(x, y);
            }
            else {
                ctx.lineTo(x, y);
                ctx.stroke();
            }
        });
        canvas.addEventListener("mouseleave", () => {
            painting = false;
        });
    }, [])

    return (
        <canvas id={'canvas'}/>
    )
}

export default Canvas;