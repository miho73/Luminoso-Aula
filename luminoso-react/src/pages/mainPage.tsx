import axios from "axios";
import Canvas from "./canvas";
import {useState} from "react";

import '../styles/button.scss';

function MainPage(props: {redirect: Function}) {
    const [working, setWorking] = useState(false);

    function reset() {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    function send() {
        setWorking(true);

        let grayscale = getImageGrayscale();

        axios.post('/api/ml/predict', {
            image: grayscale[2],
            width: grayscale[0],
            height: grayscale[1]
        }).then((response) => {
            let prediction = response.data['pred'];

            let house = getHouse(prediction);
            if(house != 5) props.redirect(house+1);
        }).catch((error) => {
            console.log(error);
        }).finally(() => {
            setWorking(false);
        });
    }

    return (
        <main className={'index'}>
            <img className={'title'} src={'https://1000logos.net/wp-content/uploads/2021/04/Hogwarts-Logo.png'} alt={'Harry Potter'}/>
            <div className={'draw'}>
                <Canvas/>
            </div>
            <div className={'control'}>
                <div className={'button-container'}>
                    <span className='mas'></span>
                    <button className={'fst'} type='button' onClick={reset} disabled={working}>초기화</button>
                </div>
                <div className="button-container">
                    <span className="mas"></span>
                    <button className={'sec'} type="button" onClick={send} disabled={working}>확인</button>
                </div>
            </div>
        </main>
    )
}

function getImageGrayscale() {
    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

    let imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let grayscale: number[] = [];
    for(let i = 3; i < imgData.data.length; i += 4) {
        grayscale.push(imgData.data[i]);
    }
    return [canvas.width, canvas.height, grayscale];
}

function getHouse(prediction: number) {
    switch (prediction) {
        case 8:
            return 0; // Gryffindor
        case 5:
        case 6:
            return 1; // Slytherin
        case 0:
        case 1:
        case 2:
            return 2; // Ravenclaw
        case 3:
        case 7:
            return 3; // Hufflepuff
        case 4:
            return 4; // Death Eater
        default:
            return 5; // Unknown
    }
}

export default MainPage;