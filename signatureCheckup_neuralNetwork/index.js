import { readdirSync, readFileSync } from 'fs';
import { node, tensor1d, sequential, layers } from '@tensorflow/tfjs-node';


function loadImages(directory) {
    const files = readdirSync(directory);
    return files.map(file => {
        const img = readFileSync(`${directory}/${file}`);
        const tensor = node.decodeImage(img, 1); 
        return tensor.resizeBilinear([28, 28]).toFloat().div(255.0);
    });
}
// To load photos of the signatures
const signatures = [
    loadImages('//'), // Signature 1
    loadImages('//'), // Signature 2
    loadImages('//'), // Signature 3
    loadImages('//'), // Fake Signature
];

// (0 for genuine, 1 for fake)
const labels = [0, 0, 0, 1]; // 0 for genuine, 1 for fake
const labelsTensor = tensor1d(labels);
// Definining the model
const model = sequential();
model.add(layers.flatten({ inputShape: [28, 28, 1] }));
model.add(layers.dense({ units: 128, activation: 'relu' }));
model.add(layers.dense({ units: 2, activation: 'softmax' }));

// To compile the model
model.compile({
    optimizer: 'adam',
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
});

// Train the model
async function trainModel() {
    const batchSize = 32;
    const epochs = 10;
    return await model.fit(signatures, labelsTensor, {
        batchSize,
        epochs,
        shuffle: true,
    });
}

trainModel().then(history => {
    console.log('Training history:', history.history);
    //To save the trained model
    model.save('C:/Users/joaop/OneDrive/Documentos/signatureCheckup_neuralNetwork');
});

