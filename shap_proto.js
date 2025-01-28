async function calculateShapleyValues(model, baseline, inputData, numSamples = 100) {
    const featureCount = inputData.length;
    let shapleyValues = Array(featureCount).fill(0);

    // Helper function to shuffle array
    function shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    // Iterate over each feature
    for (let i = 0; i < featureCount; i++) {
        let marginalContributions = [];

        for (let sample = 0; sample < numSamples; sample++) {
            // Generate a random permutation of feature indices
            const permutedIndices = shuffle([...Array(featureCount).keys()]);
            let coalition = [];
            let withoutFeature = [...baseline];
            let withFeature = [...baseline];

            for (const index of permutedIndices) {
                if (index === i) {
                    withFeature[i] = inputData[i];
                    break;
                }
                coalition.push(index);
                withoutFeature[index] = inputData[index];
                withFeature[index] = inputData[index];
            }

            // Predict outputs for coalitions with and without the feature
            const outputWith = (await model.predict(tf.tensor2d([withFeature], [1, featureCount]))).arraySync()[0][0];
            const outputWithout = (await model.predict(tf.tensor2d([withoutFeature], [1, featureCount]))).arraySync()[0][0];

            const marginal = outputWith - outputWithout;
            marginalContributions.push(marginal);
        }

        // Calculate the average marginal contribution for this feature
        shapleyValues[i] = marginalContributions.reduce((sum, x) => sum + x, 0) / numSamples;
    }

    return shapleyValues;
}

// Example usage:
(async () => {
    const inputData = [95, 30, 10, 3, 2]; // Example input data
    const baseline = [0, 0, 0, 0, 0]; // Baseline values (e.g., all zeros)
    const numSamples = 100; // Number of samples for approximation

    const shapleyValues = await calculateShapleyValues(model, baseline, inputData, numSamples);

    console.log("Shapley Values for each feature:");
    console.log(shapleyValues);

    const featureNames = [
        'Payment History',
        'Credit Utilization',
        'Credit History Length',
        'Types of Credit',
        'New Credit Inquiries',
    ];

    featureNames.forEach((name, i) => {
        console.log(`${name}: ${shapleyValues[i].toFixed(4)}`);
    });
})();
