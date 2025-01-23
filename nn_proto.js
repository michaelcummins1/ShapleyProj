import * as tf from '@tensorflow/tfjs';

// Define a more realistic dataset for credit score prediction
// Features: Payment history (%), Credit utilization (%), Credit history length (years), Types of credit, New credit inquiries
// Labels: Ground truth credit scores (synthetic for demonstration purposes)
const data = [
    [95, 30, 10, 3, 2, 750],
    [85, 45, 8, 2, 5, 700],
    [65, 70, 5, 1, 10, 600],
    [90, 40, 15, 4, 3, 720],
    [80, 50, 7, 2, 7, 680],
    [99, 20, 20, 5, 1, 800],
    [70, 60, 4, 1, 9, 640],
    [88, 35, 12, 3, 2, 730],
    [55, 85, 3, 1, 12, 580],
    [92, 25, 18, 4, 4, 760],
];

// Split features and labels
const features = tf.tensor2d(data.map(d => d.slice(0, -1)), [data.length, 5]);
const labels = tf.tensor2d(data.map(d => [d[5]]), [data.length, 1]);

// Define a simpler linear regression model without a hidden layer
const model = tf.sequential();
model.add(
    tf.layers.dense({
        inputShape: [5], // 5 features
        units: 1, // 1 output
        useBias: true, // Include bias term
    })
);

// Compile the model with mean squared error loss and SGD optimizer
model.compile({
    loss: 'meanSquaredError',
    optimizer: tf.train.sgd(0.0001),
});

// Train the model
(async () => {
    const epochs = 2000;
    await model.fit(features, labels, {
        epochs,
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if ((epoch + 1) % 200 === 0) {
                    console.log(
                        `Epoch [${epoch + 1}/${epochs}], Loss: ${logs.loss.toFixed(4)}`
                    );
                }
            },
        },
    });

    // Extract weights and biases of the trained model
    const [weights, bias] = model.getWeights();
    const weightsArray = await weights.data();
    const biasValue = (await bias.data())[0];

    // Feature names for clarity
    const featureNames = [
        'Payment History',
        'Credit Utilization',
        'Credit History Length',
        'Types of Credit',
        'New Credit Inquiries',
    ];

    // Print the equation for interpretability
    console.log('\nEquation for Feature Influence on Credit Score:');
    const equationTerms = [];
    weightsArray.forEach((weight, i) => {
        equationTerms.push(`(${weight.toFixed(4)}) * ${featureNames[i]}`);
    });
    const equation = `${equationTerms.join(' + ')} + Bias(${biasValue.toFixed(4)})`;
    console.log('Credit Score Estimate = ' + equation);

    // Compute percentage contribution of each feature
    const absoluteWeights = weightsArray.map(Math.abs);
    const totalWeight = absoluteWeights.reduce((sum, w) => sum + w, 0);
    const percentContributions = absoluteWeights.map(
        w => (w / totalWeight) * 100
    );

    // Print the percentage contribution of each feature
    console.log('\nPercentage Contribution of Each Feature to the Credit Score Estimate:');
    featureNames.forEach((name, i) => {
        console.log(`${name}: ${percentContributions[i].toFixed(2)}%`);
    });
})();
